from typing import Callable, Dict, List, Optional, Tuple, Union
from accelerate import Accelerator
from transformers.trainer import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    EvalPrediction,
    TrainingArguments,
    unwrap_model,
    nn,
    Trainer,
    Dataset,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    _is_peft_model,
)
from textbrewer import (
    DistillationConfig,
    KD_LOSS_MAP,
    MATCH_LOSS_MAP,
    FEATURES,
    PROJ_MAP,
)
from textbrewer.distiller_utils import select_logits_with_mask, probability_shift_
from transformers import DataCollatorForLanguageModeling
from peft import PeftModel

import torch

class KDTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        teacher_model: Union[PreTrainedModel, nn.Module],
        distill_config: DistillationConfig,
        args: TrainingArguments,
        data_collator: DataCollatorForLanguageModeling,
        train_dataset: Optional[Dataset],
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        model_init: Optional[Callable[[], "PreTrainedModel"]] = None,
        compute_metrics: Optional[Callable[["EvalPrediction"], Dict]] = None,
        callbacks: Optional[List["TrainerCallback"]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (
            None,
            None,
        ),
        preprocess_logits_for_metrics: Optional[
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        ] = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self.teacher_accelerator = Accelerator()
        self.teacher_model = self.teacher_accelerator.prepare(teacher_model)
        self.teacher_model.eval()

        self.d_config = distill_config
        self.kd_loss = KD_LOSS_MAP[self.d_config.kd_loss_type]
        self.projs = []
        self.projs_group = []
        for im in self.d_config.intermediate_matches:
            if im.proj is not None:
                projection = im.proj[0]
                dim_in = im.proj[1]
                dim_out = im.proj[2]
                self.projs_group.append(im.proj[3])
                self.projs.append(PROJ_MAP[projection](dim_in, dim_out))
                self.projs[-1].to(self.t_config.device)
            else:
                self.projs.append(None)
                self.projs_group.append(None)

        # 目前来看模型的初始化应该在__wrap_model中即可

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        total_loss = 0

        with torch.no_grad():
            results_T = self.teacher_model(**inputs, output_hidden_states=True)

        # Below is copied from transformers.Trainer.compute_loss() in transformer@2f12e40
        if self.d_config.hard_label_weight > 0:
            if self.label_smoother is not None and "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            outputs = model(**inputs, output_hidden_states=True)
            # Save past state if it exists
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index]

            if labels is not None:
                unwrapped_model = unwrap_model(model)
                if _is_peft_model(unwrapped_model):
                    model_name = unwrapped_model.base_model.model._get_name()
                else:
                    model_name = unwrapped_model._get_name()
                if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                    # 只有causal lm才需要shift label
                    loss = self.label_smoother(outputs, labels, shift_labels=True)
                else:
                    loss = self.label_smoother(outputs, labels)
            else:
                if isinstance(outputs, dict) and "loss" not in outputs:
                    raise ValueError(
                        "The model did not return a loss from the inputs, only the following keys: "
                        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                    )
                # We don't use .loss here since the model may return tuples instead of ModelOutput.
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                total_loss += loss * self.d_config.hard_label_weight

        results_S = outputs
        # KD loss copied from textbrewer general distiller without custom match and loss dict
        total_kd_loss = 0
        if "logits_mask" in results_S:
            masks_list_S = results_S["logits_mask"]
            logits_list_S = select_logits_with_mask(
                logits_list_S, masks_list_S
            )  # (mask_sum, num_of_class)
        if "logits_mask" in results_T:
            masks_list_T = results_T["logits_mask"]
            logits_list_T = select_logits_with_mask(logits_list_T, masks_list_T)
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
            for l_T, l_S in zip(results_T.logits, results_S.logits):
                if self.d_config.temperature_scheduler is not None:
                    temperature = self.d_config.temperature_scheduler(
                        l_S, l_T, self.d_config.temperature
                    )
                else:
                    temperature = self.d_config.temperature
                total_kd_loss += self.kd_loss(l_S, l_T, temperature)
        total_loss += total_kd_loss * self.d_config.kd_loss_weight

        inters_T = {feature: results_T.get(feature, []) for feature in FEATURES}
        inters_S = {feature: results_S.get(feature, []) for feature in FEATURES}
        inputs_mask_T = results_T.get("inputs_mask", None)
        inputs_mask_S = results_S.get("inputs_mask", None)
        for ith, inter_match in enumerate(self.d_config.intermediate_matches):
            layer_T = inter_match.layer_T
            layer_S = inter_match.layer_S
            feature = inter_match.feature
            loss_type = inter_match.loss
            match_weight = inter_match.weight
            match_loss = MATCH_LOSS_MAP[loss_type]

            if type(layer_S) is list and type(layer_T) is list:
                inter_S = [inters_S[feature][s] for s in layer_S]
                inter_T = [inters_T[feature][t] for t in layer_T]
                name_S = "-".join(map(str, layer_S))
                name_T = "-".join(map(str, layer_T))
                if self.projs[ith]:
                    # inter_T = [self.projs[ith](t) for t in inter_T]
                    inter_S = [self.projs[ith](s) for s in inter_S]
            else:
                inter_S = inters_S[feature][layer_S]
                inter_T = inters_T[feature][layer_T]
                name_S = str(layer_S)
                name_T = str(layer_T)
                if self.projs[ith]:
                    # inter_T = self.projs[ith](inter_T)
                    inter_S = self.projs[ith](inter_S)
            intermediate_loss = match_loss(inter_S, inter_T, mask=inputs_mask_S)
            total_loss += intermediate_loss * match_weight

        # TODO multi gpu loss?
        if "losses" in results_S:
            total_hl_loss = 0
            for loss in results_S["losses"]:
                # in case of multi-GPU
                total_hl_loss += loss.mean()
            total_loss += total_hl_loss * self.d_config.hard_label_weight

        return (total_loss, outputs) if return_outputs else loss
