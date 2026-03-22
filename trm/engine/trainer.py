import datetime
import logging
import os
import time
import gc
import torch
import torch.distributed as dist

from trm.data import make_data_loader
from trm.utils.comm import get_world_size, synchronize
from trm.utils.metric_logger import MetricLogger
from trm.engine.inference import inference
from ..utils.comm import is_main_process


def reduce_loss(loss):
    world_size = get_world_size()
    if world_size < 2:
        return loss
    with torch.no_grad():
        dist.reduce(loss, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            loss /= world_size
    loss = loss.item()
    return loss


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    param_dict,
    max_norm=5
):

    logger = logging.getLogger("trm.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_epoch = cfg.SOLVER.MAX_EPOCH

    model.train()
    start_training_time = time.time()
    end = time.time()
    max_iteration = len(data_loader)
    writer_count = 0

    for epoch in range(arguments["epoch"], max_epoch + 1):
        rest_epoch_iteration = (max_epoch - epoch) * max_iteration
        arguments["epoch"] = epoch
        if get_world_size() > 1:
            data_loader.batch_sampler.sampler.set_epoch(epoch)
        if epoch <= cfg.SOLVER.FREEZE_BERT:
            for param in param_dict['bert']:
                param.requires_grad_(False)
        else:
            for param in param_dict['bert']:
                param.requires_grad_(True)
        logger.info("Start epoch {}. base_lr={:.1e}, bert_lr={:.1e}, bert.requires_grad={}".format(epoch, optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], str(param_dict['bert'][0].requires_grad)))
        if epoch <= cfg.SOLVER.ONLY_IOU:
            logger.info("Using all losses")
        else:
            logger.info("Using only bce loss")
        for iteration, (batches, idx) in enumerate(data_loader):
            writer_count += 1
            iteration += 1
            batches = batches.to(device)
            optimizer.zero_grad()
            contr_weight = cfg.MODEL.TRM.LOSS.CONTRASTIVE_WEIGHT
            consis_weight = cfg.MODEL.TRM.LOSS.CONSIS_WEIGHT
            exc_weight = cfg.MODEL.TRM.LOSS.EXC_WEIGHT
            loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg, scoremap_loss_exc = model(batches, cur_epoch=epoch)
            # print(loss_vid, loss_sent, loss_iou_stnc, loss_iou_phrase, scoremap_loss_pos, scoremap_loss_neg )
            loss_vid, loss_sent = loss_vid * contr_weight, loss_sent * contr_weight
            scoremap_loss_pos, scoremap_loss_neg = scoremap_loss_pos * consis_weight, scoremap_loss_neg * consis_weight,
            scoremap_loss_exc = scoremap_loss_exc * exc_weight
            meters.update(loss_vid=loss_vid.detach(), loss_sent=loss_sent.detach(), loss_iou_stnc=loss_iou_stnc.detach(), loss_iou_phrase=loss_iou_phrase.detach(), scoremap_loss_pos=scoremap_loss_pos.detach(), scoremap_loss_neg=scoremap_loss_neg.detach(), scoremap_loss_exc=scoremap_loss_exc.detach())
            loss = 0
            if epoch <= cfg.SOLVER.ONLY_IOU:
                loss += loss_iou_phrase + loss_iou_stnc + (scoremap_loss_pos + scoremap_loss_neg)*0.5 + scoremap_loss_exc
                loss += loss_sent + loss_vid
            else:
                loss += loss_iou_phrase + loss_iou_stnc + (scoremap_loss_pos + scoremap_loss_neg)*0.5 + scoremap_loss_exc
                loss += (loss_sent + loss_vid) * 0.01
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (max_iteration - iteration + rest_epoch_iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if iteration % 10 == 0 or iteration == max_iteration:
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "epoch: {epoch}/{max_epoch}",
                            "iteration: {iteration}/{max_iteration}",
                            "{meters}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        epoch=epoch,
                        max_epoch=max_epoch,
                        iteration=iteration,
                        max_iteration=max_iteration,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            gc.collect()

        scheduler.step()
        if checkpoint_period != -1 and epoch % checkpoint_period == 0:
            checkpointer.save(f"{cfg.MODEL.TRM.FEAT2D.NAME}_model_{epoch}e", **arguments)

        if data_loader_val is not None and test_period > 0 and epoch % test_period == 0 and epoch >= cfg.SOLVER.SKIP_TEST:
            synchronize()
            torch.cuda.empty_cache()
            result_dict = inference(
                cfg,
                model,
                data_loader_val,
                dataset_name=cfg.DATASETS.TEST,
                nms_thresh=cfg.TEST.NMS_THRESH,
                device=cfg.MODEL.DEVICE,
            )
            synchronize()
            model.train()
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iteration)
        )
    )
