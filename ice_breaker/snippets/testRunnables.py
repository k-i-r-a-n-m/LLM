from langchain_core.runnables import RunnableLambda

import random

import decorators as mp

def add_one(x: int) -> int:
    return x + 1


def buggy_double(y: int) -> int:
    '''Buggy code that will fail 70% of the time'''
    if random.random() > 0.3:
        print('This code failed, and will probably be retried!')
        raise ValueError('Triggered buggy code')
    return y * 2

def outputParsed(z):
    return 'Answer: '+str(z)

sequence = (
    RunnableLambda(add_one) |
    RunnableLambda(buggy_double).with_retry( # Retry on failure
        stop_after_attempt=10,
        wait_exponential_jitter=False
    ) | RunnableLambda(outputParsed)
)

print(sequence.input_schema.schema()) # Show inferred input schema
print(sequence.output_schema.schema()) # Show inferred output schema
# print(sequence.invoke(2)) # invoke the sequence (note the retry above!!)

print(outputParsed(3))


mp.welcome()
