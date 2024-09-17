import dataclasses
import json
import os
from typing import Dict, List, Union, Any
from importlib import resources


import typer
from rich import print

from instruction_following_eval import instructions_registry

DEFAULT_FILE = resources.files("instruction_following_eval") / "data/input_data.jsonl"



@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: List[str]
    prompt: str
    kwargs: List[Dict[str, Any]]

@dataclasses.dataclass
class OutputExample:
    instruction_id_list: List[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: List[bool]

def mean(numbers: List[float]) -> float:
    """Calculate the mean of a list of numbers."""
    return sum(numbers) / len(numbers) if numbers else 0.0

def instruction_mean(results: List[OutputExample]) -> float:
    """Calculate the mean accuracy across all instructions."""
    all_instructions = [inst for result in results for inst in result.follow_instruction_list]
    return mean(all_instructions)

def get_examples(input_jsonl_filename: str = DEFAULT_FILE) -> List[Dict[str, Any]]:
    """Read inputs from jsonl."""
    inputs = []
    with open(input_jsonl_filename, "r") as f:
        for l in f:
            example = json.loads(l)
            inputs.append(example)
    return inputs

def dict_to_input_example(example: Dict[str, Any]) -> InputExample:
    """Convert a dictionary to an InputExample."""
    return InputExample(
        key=example["key"],
        instruction_id_list=example["instruction_id_list"],
        prompt=example["prompt"],
        kwargs=example["kwargs"]
    )

def test_instruction_following(example: Union[Dict[str, Any], InputExample], response: str, strict: bool = True) -> OutputExample:
    """Tests response to see if instructions are followed."""
    if isinstance(example, dict):
        example = dict_to_input_example(example)

    is_following_list = []

    if not strict:
        responses = [
            response,
            response.replace("*", ""),
            "\n".join(response.split("\n")[1:]).strip(),
            "\n".join(response.split("\n")[:-1]).strip(),
            "\n".join(response.split("\n")[1:-1]).strip(),
        ]
        responses += [r.replace("*", "") for r in responses[2:]]
    else:
        responses = [response]

    for index, instruction_id in enumerate(example.instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**example.kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=example.prompt)

        is_following = any(r.strip() and instruction.check_following(r) for r in responses)
        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=example.instruction_id_list,
        prompt=example.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )

def evaluate_instruction_following(examples: List[Dict[str, Any]], responses: List[str]) -> Dict[str, float]:
    if len(examples) != len(responses):
        raise ValueError("The number of examples and responses must be the same.")

    input_examples = [dict_to_input_example(ex) for ex in examples]
    strict_results = [test_instruction_following(ex, resp, strict=True) for ex, resp in zip(input_examples, responses)]
    loose_results = [test_instruction_following(ex, resp, strict=False) for ex, resp in zip(input_examples, responses)]

    return {
        "prompt_level_strict_accuracy": mean([r.follow_all_instructions for r in strict_results]),
        "inst_level_strict_accuracy": instruction_mean(strict_results),
        "prompt_level_loose_accuracy": mean([r.follow_all_instructions for r in loose_results]),
        "inst_level_loose_accuracy": instruction_mean(loose_results),
    }

def print_report(results: Dict[str, float]):
    """Prints a report on accuracy scores."""
    print("Evaluation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

def main(
    input_data: str = typer.Option(..., help="Path to input data"),
    input_response_data: str = typer.Option(..., help="Path to input response data"),
    output_dir: str = typer.Option(..., help="Output directory for inference and eval results")
):
    examples = get_examples(input_data)
    
    # Read responses
    with open(input_response_data, "r") as f:
        responses = [json.loads(line)["response"] for line in f]

    results = evaluate_instruction_following(examples, responses)

    # Print the report
    print_report(results)

    # Save results to a file
    output_file = os.path.join(output_dir, "evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    typer.run(main)
