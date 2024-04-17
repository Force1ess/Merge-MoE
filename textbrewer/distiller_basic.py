from .distiller_utils import *


class BasicDistiller(AbstractDistiller):
    """
    Performs **single-teacher single-task** distillation, provides basic distillation strategies.

    Args:
        train_config (:class:`TrainingConfig`): training configuration.
        distill_config (:class:`DistillationConfig`): distillation configuration.
        model_T (:class:`torch.nn.Module`): teacher model.
        model_S (:class:`torch.nn.Module`): student model.
        adaptor_T (Callable): teacher model's adaptor.
        adaptor_S (Callable): student model's adaptor.

    The roles of `adaptor_T` and `adaptor_S` are explained in :py:func:`adaptor`.

    """

    def __init__(
        self, train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S
    ):
        super(BasicDistiller, self).__init__(
            train_config, distill_config, model_T, model_S, adaptor_T, adaptor_S
        )

    def save_and_callback(self, global_step, step, epoch, callback):
        # 所有进程等待直到到达这个点
        torch.distributed.barrier()

        if self.rank == 0:  # 仅 rank 0 执行保存操作
            logger.info(
                f"Saving at global step {global_step}, epoch step {step + 1} epoch {epoch+1}"
            )
            self.model_S.module.save_pretrained(
                self.t_config.output_dir + f"_step{global_step}"
            )

        # 所有进程再次同步，确保模型已保存，然后再继续
        torch.distributed.barrier()

        # 所有进程执行回调
        if callback is not None:
            logger.info("Running callback function...")
            callback(model=self.model_S, step=global_step)
            self.model_S.train()

    def write_loss(self, total_loss, writer_step, losses_dict=None):
        if self.rank == 0:
            cpu_total_loss = total_loss.cpu().item()
            self.tb_writer.add_scalar("scalar/total_loss", cpu_total_loss, writer_step)
            if losses_dict is not None:
                for name, loss in losses_dict.items():
                    cpu_loss = loss.cpu().item()
                    self.tb_writer.add_scalar(f"scalar/{name}", cpu_loss, writer_step)

    def initialize_training(
        self, optimizer, scheduler_class, scheduler_args, scheduler
    ):
        # update optimizer for projection layer (used in GeneralDistiller)
        if hasattr(self, "projs"):
            for proj, proj_group in zip(self.projs, self.projs_group):
                if proj is not None:
                    assert isinstance(proj, nn.Module)
                    optimizer.add_param_group(
                        {**{"params": proj.parameters()}, **proj_group}
                    )

        if hasattr(self, "has_custom_matches") and self.has_custom_matches:
            for proj_func, proj_group in zip(
                self.custom_matches_cache["match_proj_funcs"],
                self.custom_matches_cache["match_proj_groups"],
            ):
                if isinstance(proj_func, nn.Module):
                    optimizer.add_param_group(
                        {**{"params": proj_func.parameters()}, **proj_group}
                    )

        logger.debug("Optimizer param group: ")
        logger.debug(
            f"{[[s.shape for s in g['params']] for g in optimizer.param_groups]}"
        )

        # update scheduler
        if scheduler_class is not None:
            # overwrite scheduler
            scheduler = scheduler_class(**{"optimizer": optimizer}, **scheduler_args)

        tqdm_disable = None if self.rank == 0 else True
        return optimizer, scheduler, tqdm_disable

    def train_with_num_epochs(
        self,
        optimizer,
        scheduler,
        tqdm_disable,
        dataloader,
        max_grad_norm,
        num_epochs,
        callback,
        batch_postprocessor,
        **args,
    ):
        train_steps_per_epoch = (
            len(dataloader) // self.t_config.gradient_accumulation_steps
        )
        total_global_steps = train_steps_per_epoch * num_epochs
        pbar = tqdm(total=total_global_steps, disable=tqdm_disable)
        print_every = train_steps_per_epoch // self.print_freq
        if print_every == 0:
            print_every = train_steps_per_epoch
        checkpoints = [
            int(train_steps_per_epoch * ci / self.t_config.ckpt_frequency)
            for ci in range(self.t_config.ckpt_frequency)
        ]
        logger.info(f"Training steps per epoch: {train_steps_per_epoch}")
        logger.info(f"Checkpoints(step): {checkpoints}")

        global_step = 0
        writer_step = 0

        for current_epoch in range(int(num_epochs)):
            if self.local_rank != -1 and hasattr(dataloader, "sampler"):
                dataloader.sampler.set_epoch(
                    current_epoch
                )  # In distributed mode, calling the set_epoch method is needed to make shuffling work;
            logger.info(f"Epoch {current_epoch+1}")
            optimizer.zero_grad()
            logger.info(f"Length of current epoch in forward batch: {len(dataloader)}")
            for step, batch in enumerate(dataloader):
                if (
                    self.d_config.is_caching_logits is False
                    and batch_postprocessor is not None
                ):
                    batch = batch_postprocessor(batch)
                total_loss, losses_dict = self.train_on_batch(batch, args)

                self.write_loss(total_loss, writer_step, losses_dict)
                writer_step += 1

                total_loss /= self.t_config.gradient_accumulation_steps
                total_loss.backward()

                if (step + 1) % self.t_config.gradient_accumulation_steps == 0:
                    if max_grad_norm > 0:
                        if self.t_config.fp16:
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(optimizer), max_grad_norm
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model_S.parameters(), max_grad_norm
                            )
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if self.d_config.kd_loss_weight_scheduler is not None:
                        self.d_config.kd_loss_weight = (
                            self.d_config.kd_loss_weight_scheduler(
                                global_step / total_global_steps
                            )
                        )
                    if self.d_config.hard_label_weight_scheduler is not None:
                        self.d_config.hard_label_weight = (
                            self.d_config.hard_label_weight_scheduler(
                                global_step / total_global_steps
                            )
                        )

                    if (global_step) % print_every == 0:
                        logger.info(f"Global step: {global_step}, epoch step:{step+1}")
                    if (global_step % train_steps_per_epoch in checkpoints) and (
                        (current_epoch + 1) % self.t_config.ckpt_epoch_frequency == 0
                        or current_epoch + 1 == num_epochs
                    ):
                        self.save_and_callback(
                            global_step, step, current_epoch, callback
                        )
                pbar.update(1)

            logger.info(f"Epoch {current_epoch+1} finished")

    def train(
        self,
        optimizer,
        dataloader,
        num_epochs=None,
        scheduler_class=None,
        scheduler_args=None,
        scheduler=None,
        max_grad_norm=-1.0,
        num_steps=None,
        callback=None,
        batch_postprocessor=None,
        **args,
    ):
        """
        trains the student model.

        Args:
            optimizer: optimizer.
            dataloader: dataset iterator.
            num_epochs (int): number of training epochs.
            num_steps (int): number of training steps. If it is not None, distiller will ignore `num_epochs` and trains for `num_steps`, and dataloader can have an unkonwn size, i.e., has no `__len__` attribute. Dataloader will be cycled automatically after iterating over the whole dataset.
            callback (Callable): function called after each epoch, can be None. It is called as ``callback(model=self.model_S, step = global_step)``. It can be used to evaluate the model at each checkpoint.
            batch_postprocessor (Callable): a function for post-processing batches. It should take a batch and return a batch. Its output is fed to the models and adaptors.
            scheduler_class (class): the class of the scheduler to be constructed.
            scheduler_args (dict): arguments (excluding `optimizer`) passed to the `scheduler_class` to construct the scheduler object. See the example below.
            scheduler (deprecated): used to adjust learning rate, optional, can be None, is deprecated in favor of `scheduler_class` and `scheduler_args`.
            max_grad_norm (float): Maximum norm for the gradients (-1 means no clipping). Default: -1.0
            **args: additional arguments fed to the model.
        Note:
            * If the batch is a list or tuple, model is called as: ``model(*batch, **args)``. Make sure the order of elements in the batch matches their order in ``model.forward``.
            * If the batch is a dict, model is called as: ``model(**batch,**args)``. Make sure the keys of the batch match the arguments of the ``model.forward``.
        Note:
            If you want to provide a lr scheduler, DON'T USE `scheduler` , use `scheduler_class` and `scheduler_args` instead. Example:

            .. code-block::

                from transformers import get_linear_schedule_with_warmup
                distiller.train(optimizer, scheduler_class = get_linear_schedule_with_warmup, scheduler_args= {'num_warmup_steps': 100, 'num_training_steps': 1000})
        """
        optimizer, scheduler, tqdm_disable = self.initialize_training(
            optimizer, scheduler_class, scheduler_args, scheduler
        )

        assert not num_epochs is None
        self.train_with_num_epochs(
            optimizer,
            scheduler,
            tqdm_disable,
            dataloader,
            max_grad_norm,
            num_epochs,
            callback,
            batch_postprocessor,
            **args,
        )

    def train_on_batch(self, batch, args):
        if self.d_config.is_caching_logits is False:
            (teacher_batch, results_T), (
                student_batch,
                results_S,
            ) = get_outputs_from_batch(
                batch, self.t_config.device, self.model_T, self.model_S, args
            )
            results_T = post_adaptor(self.adaptor_T(teacher_batch, results_T))
            results_S = post_adaptor(self.adaptor_S(student_batch, results_S))
        else:
            batch, cached_logits = batch
            _, (student_batch, results_S) = get_outputs_from_batch(
                batch,
                self.t_config.device,
                self.model_T,
                self.model_S,
                args,
                no_teacher_forward=True,
            )

            results_S = post_adaptor(self.adaptor_S(student_batch, results_S))
            results_T = {
                "logits": [logits.to(self.t_config.device) for logits in cached_logits]
            }

            if "logits_mask" in results_S:
                results_T["logits_mask"] = results_S["logits_mask"]

        total_loss, losses_dict = self.compute_loss(results_S, results_T)

        return total_loss, losses_dict

    def compute_loss(self, results_S, results_T):
        total_loss = 0
        losses_dict = dict()
        logits_list_T = results_T["logits"]  # list of tensor
        logits_list_S = results_S["logits"]  # list of tensor

        if "logits_mask" in results_S:
            masks_list_S = results_S["logits_mask"]
            logits_list_S = select_logits_with_mask(
                logits_list_S, masks_list_S
            )  # (mask_sum, num_of_class)
        if "logits_mask" in results_T:
            masks_list_T = results_T["logits_mask"]
            logits_list_T = select_logits_with_mask(
                logits_list_T, masks_list_T
            )  # (mask_sum, num_of_class)

        total_kd_loss = 0
        if self.d_config.probability_shift is True:
            labels_list = results_S["labels"]
            for l_T, l_S, labels in zip(logits_list_T, logits_list_S, labels_list):
                l_T = probability_shift_(l_T, labels)
                if self.d_config.temperature_scheduler is not None:
                    temperature = self.d_config.temperature_scheduler(
                        l_S, l_T, self.d_config.temperature
                    )
                else:
                    temperature = self.d_config.temperature
                total_kd_loss += self.kd_loss(l_S, l_T, temperature)
        else:
            for l_T, l_S in zip(logits_list_T, logits_list_S):
                if self.d_config.temperature_scheduler is not None:
                    temperature = self.d_config.temperature_scheduler(
                        l_S, l_T, self.d_config.temperature
                    )
                else:
                    temperature = self.d_config.temperature
                total_kd_loss += self.kd_loss(l_S, l_T, temperature)
        total_loss += total_kd_loss * self.d_config.kd_loss_weight
        losses_dict["unweighted_kd_loss"] = total_kd_loss

        if "losses" in results_S:
            total_hl_loss = 0
            for loss in results_S["losses"]:
                # in case of multi-GPU
                total_hl_loss += loss.mean()
            total_loss += total_hl_loss * self.d_config.hard_label_weight
            losses_dict["unweighted_hard_label_loss"] = total_hl_loss
        return total_loss, losses_dict

    def cache_logits(self, batch, args, batch_postprocessor):
        if batch_postprocessor is not None:
            batch = batch_postprocessor(batch)
        batch = move_to_device(batch, self.t_config.device)
        with torch.no_grad():
            if type(batch) is dict:
                results_T = self.model_T(**batch, **args)
            else:
                results_T = self.model_T(*batch, **args)
        results_T = post_adaptor(self.adaptor_T(batch, results_T))

        self.logits_cache.append(
            [batch, [logits.to("cpu") for logits in results_T["logits"]]]
        )
