# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from typing import Union, Any, Literal
import json
from importlib import resources

import pandas as pd
from pydantic import validate_call

from . import instructions_registry


def default_examples():
    inputs = []
    data_path = resources.files("instruction_following_eval") / "data/input_data.jsonl"
    with data_path.open("r") as f:
        for line in f:
            example = json.loads(line)
            # TODO: fix this
            if "combination:repeat_prompt" in example["instruction_id_list"]:
                continue
            inputs.append(example)
    return inputs


@dataclasses.dataclass
class InstructionResult:
    key: int
    instruction_id_list: list[str]
    prompt: str
    response: str
    kwargs: list[dict[str, Any]]


@dataclasses.dataclass
class InstructionEval:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]

    def as_logs(self):
        logs = []
        for instruction, follow in zip(
            self.instruction_id_list, self.follow_instruction_list
        ):
            group, category = instruction.split(":")
            logs.append(
                {
                    "group": group,
                    "category": category,
                    "follow": follow,
                }
            )
        return logs


Logs = list[dict[str, Any]]


@validate_call
def instruction_following_eval(
    examples: list[InstructionResult],
    strict: bool = True,
    aggregate: bool = True,
) -> Union[dict[str, float], Logs]:
    eval_fn = {
        True: test_instruction_following_strict,
        False: test_instruction_following_loose,
    }[strict]

    logs = sum((eval_fn(example).as_logs() for example in examples), start=[])

    if not aggregate:
        return logs

    results = pd.DataFrame.from_records(logs)
    by_category = results.groupby(["group", "category"], as_index=False).follow.mean()
    acc_dict = by_category.set_index('category').follow.to_dict()
    by_group = by_category.groupby(["group"], as_index=False).follow.mean()
    acc_dict['average'] = by_group.follow.mean()
    return acc_dict


@validate_call
def test_instruction_following_strict(example: InstructionResult) -> InstructionEval:
    """Tests response to see if instructions are followed."""
    response = example.response
    instruction_list = example.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**example.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=example.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return InstructionEval(
        instruction_id_list=example.instruction_id_list,
        prompt=example.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


@validate_call
def test_instruction_following_loose(example: InstructionResult) -> InstructionEval:
    """Tests response for an upper bound for following instructions."""
    response = example.response
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = example.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**example.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=example.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return InstructionEval(
        instruction_id_list=example.instruction_id_list,
        prompt=example.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )
