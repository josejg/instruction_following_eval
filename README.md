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

TODO


### Tests

```shell
$ python test/instructions_test.py
$ python test/instructions_util_test.py
```
