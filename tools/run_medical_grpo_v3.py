#!/usr/bin/env python
"""TRL/Transformers 0.28/5.x compatibility entrypoint for medical GRPO."""

import run_medical_grpo_v2 as compatibility


implementation = compatibility.implementation
previous_init = implementation.LearnedReward.__init__


def compatible_init(self, *args, **kwargs):
    previous_init(self, *args, **kwargs)
    self.model.config.pad_token_id = self.tokenizer.pad_token_id


implementation.LearnedReward.__init__ = compatible_init


if __name__ == "__main__":
    implementation.main()
