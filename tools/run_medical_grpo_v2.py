#!/usr/bin/env python
"""TRL 0.28 compatibility entrypoint for run_medical_grpo.py."""

import run_medical_grpo as implementation


original_init = implementation.LearnedReward.__init__


def compatible_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.__name__ = "learned_reward"


implementation.LearnedReward.__init__ = compatible_init


if __name__ == "__main__":
    implementation.main()
