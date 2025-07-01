All the commands provided here work from ROOT directory aka CoolPrompt/

To run all tests:
```
python3 -m unittest discover -s test -t .
```

To run a specific test file:
```
python3 -m unittest test.coolprompt.test_assistant
```

To run a specific TestCase (aka test class):
```
python3 -m unittest test.coolprompt.test_assistant.TestPromptTuner
```

To run a specific test method (inside of TestCase):
```
python3 -m unittest test.coolprompt.test_assistant.TestPromptTuner.test_run_reflective
```