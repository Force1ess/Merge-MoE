from typing import Callable, Dict, List, Optional, Tuple, Union
from transformers.trainer import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    EvalPrediction,
    TrainingArguments,
    nn,
    Trainer,
    Dataset,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    _is_peft_model,
    DEFAULT_PROGRESS_CALLBACK,
)
from textbrewer import (
    DistillationConfig,
    KD_LOSS_MAP,
    MATCH_LOSS_MAP,
    FEATURES,
)
from transformers import DataCollatorForLanguageModeling
from utils import rank0_print

import torch


class KDTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
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
        self.logging_steps = args.logging_steps
        self.step = 0
        self.d_config = distill_config
        self.kd_loss = KD_LOSS_MAP[self.d_config.kd_loss_type]
        self.projs = []
        self.projs_group = []
        self.pbar_handler = None
        for i in self.callback_handler.callbacks:
            if isinstance(i, DEFAULT_PROGRESS_CALLBACK):
                self.pbar_handler = i
                break

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        self.step += 1
        total_loss = 0

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
                unwrapped_model = self.accelerator.unwrap_model(model)
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
        results_T = {
            "hidden_states": [
                l.block_sparse_hidden_states.detach() for l in model.module.model.model.layers
            ]
        }
        results_T["hidden_states"].append(
            model.module.model.model.norm(results_T["hidden_states"][-1])
        )
        results_T["logits"] = model.module.lm_head(
            results_T["hidden_states"][-1]
        ).float()

        for l_T, l_S in zip(results_T["logits"], results_S.logits):
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
        total_inter_loss = 0
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
            else:
                inter_S = inters_S[feature][layer_S]
                inter_T = inters_T[feature][layer_T]
            intermediate_loss = match_loss(inter_S, inter_T, mask=None)
            total_inter_loss += intermediate_loss * match_weight
        total_loss += total_inter_loss * self.d_config.intermediate_loss_weight
        if (
            self.step % (self.logging_steps *self.args.gradient_accumulation_steps ) == 0
            and self.pbar_handler is not None
            and self.pbar_handler.training_bar is not None
        ):
            self.pbar_handler.training_bar.write(
                f"step {self.step}: [0]label Loss {loss * self.d_config.hard_label_weight} [1]logits loss {total_kd_loss * self.d_config.kd_loss_weight} [2]inter loss {total_inter_loss * self.d_config.intermediate_loss_weight}"
            )
        return (total_loss, outputs) if return_outputs else total_loss
