# Instruction Following Eval

This is an unofficial fork of [IFEval](https://github.com/josejg/google-research/tree/master/instruction_following_eval) (see [paper](https://arxiv.org/abs/2311.07911)), designed to make running instruction following eval easier. The goals of the fork are as follows: 

- Make the codebase `pip` installable & fix the dependencies
- Provide runtime bindings that can be called from your own codebase. If you are interested in a CLI based interface see the original project.

### Install 

You can install **instruction_following_eval** with `pip` directly

```shell
pip install git+https://github.com/josejg/instruction_following_eval.git
```

or manually cloning it:

```shell
git clone https://github.com/josejg/instruction_following_eval.git
cd instruction_following_eval
pip install . 
```

### Running IFEval

To run instruction following eval in your codebase you just need to import `instruction_following_eval` and provide a list of examples encoded as dictionaries with the following structure:

```python
# Example test
{
 'key': 1001,
 'instruction_id_list': ['punctuation:no_comma'],
 'prompt': 'I am planning a trip to Japan, and I would like thee to write an '
           'itinerary for my journey in a Shakespearean style. You are not '
           'allowed to use any commas in your response.',
 'kwargs': [{}],
 'response': '<MODEL RESPONSE>'
}
```

However, you can retrieve the list of examples of the [original paper](https://arxiv.org/abs/2311.07911) by calling `default_examples`.

```python 
from instruction_following_eval import default_examples, instruction_following_eval

examples = default_examples()

for example in examples:
    example['response'] = model.generate(example['prompt'])

accuracy = instruction_following_eval(examples)

print(accuracy)
```


### Tests

```shell
$ python test/instructions_test.py
$ python test/instructions_util_test.py
```
