#!/usr/bin/env python

import collections
import json
import os

from absl import app, flags, logging

from instruction_following_eval import (
    read_prompt_list,
    read_prompt_to_response_dict,
    test_instruction_following_loose,
    test_instruction_following_strict,
)

_INPUT_DATA = flags.DEFINE_string(
    "input_data", None, "path to input data", required=True
)

_INPUT_RESPONSE_DATA = flags.DEFINE_string(
    "input_response_data", None, "path to input response data", required=False
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Output directory for inference and eval results.",
    required=True,
)


def write_outputs(output_jsonl_filename, outputs):
    """Writes outputs to jsonl."""
    assert outputs
    with open(output_jsonl_filename, "w") as f:
        for o in outputs:
            f.write(
                json.dumps(
                    {
                        attr_name: getattr(o, attr_name)
                        for attr_name in [
                            name for name in dir(o) if not name.startswith("_")
                        ]
                    }
                )
            )
            f.write("\n")


def print_report(outputs):
    """Prints a report on accuracy scores."""

    prompt_total = 0
    prompt_correct = 0
    instruction_total = 0
    instruction_correct = 0

    tier0_total = collections.defaultdict(int)
    tier0_correct = collections.defaultdict(int)

    tier1_total = collections.defaultdict(int)
    tier1_correct = collections.defaultdict(int)

    for example in outputs:
        follow_instruction_list = example.follow_instruction_list
        instruction_id_list = example.instruction_id_list

        prompt_total += 1
        if all(follow_instruction_list):
            prompt_correct += 1

        instruction_total += len(instruction_id_list)
        instruction_correct += sum(follow_instruction_list)

        for instruction_id, followed_or_not in zip(
            instruction_id_list, follow_instruction_list
        ):
            instruction_id = instruction_id.split(":")[0]
            tier0_total[instruction_id] += 1
            if followed_or_not:
                tier0_correct[instruction_id] += 1

        for instruction_id, followed_or_not in zip(
            instruction_id_list, follow_instruction_list
        ):
            tier1_total[instruction_id] += 1
            if followed_or_not:
                tier1_correct[instruction_id] += 1

    print(f"prompt-level: {prompt_correct / prompt_total}")
    print(f"instruction-level: {instruction_correct / instruction_total}")
    print()
    for instruction_id in sorted(tier0_total.keys()):
        accuracy = tier0_correct[instruction_id] / tier0_total[instruction_id]
        print(f"{instruction_id} {accuracy}")
    print()
    for instruction_id in sorted(tier1_total.keys()):
        accuracy = tier1_correct[instruction_id] / tier1_total[instruction_id]
        print(f"{instruction_id} {accuracy}")


def main(argv):
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    inputs = read_prompt_list(_INPUT_DATA.value)
    prompt_to_response = read_prompt_to_response_dict(_INPUT_RESPONSE_DATA.value)

    # get instruction following results
    for func, output_file_name in [
        (test_instruction_following_strict, "eval_results_strict"),
        (test_instruction_following_loose, "eval_results_loose"),
    ]:
        logging.info("Generating %s...", output_file_name)
        outputs = []
        for inp in inputs:
            outputs.append(func(inp, prompt_to_response))
        follow_all_instructions = [o.follow_all_instructions for o in outputs]
        accuracy = sum(follow_all_instructions) / len(outputs)
        logging.info("Accuracy: %f", accuracy)

        output_file_name = os.path.join(_OUTPUT_DIR.value, output_file_name + ".jsonl")
        write_outputs(output_file_name, outputs)
        logging.info("Generated: %s", output_file_name)

        # Prints instruction following accuracy report.
        print("=" * 64)
        print(f"{output_file_name} Accuracy Scores:")
        print_report(outputs)


if __name__ == "__main__":
    app.run(main)
