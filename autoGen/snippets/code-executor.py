import tempfile
import os

from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor

# Create a temporary directory to store the code files.
temp_dir = tempfile.TemporaryDirectory()

# Create a local command line code executor.
executor = LocalCommandLineCodeExecutor(
    timeout=10,  # Timeout for each code execution in seconds.
    work_dir="./agent-code",  # Use the temporary directory to store the code files.
    execution_policies={"javascript": True}
)

# Create an agent with code executor configuration.
code_executor_agent = ConversableAgent(
    "code_executor_agent",
    llm_config=False,  # Turn off LLM for this agent.
    code_execution_config={"executor": executor},  # Use the local command line code executor.
    human_input_mode="ALWAYS",  # Always take human input for this agent for safety.
)


message_with_code_block = """This is a message with code block.
1.The code block is below:
```py
import numpy as np
import matplotlib.pyplot as plt
x = np.random.randint(0, 100, 100)
y = np.random.randint(0, 100, 100)
plt.scatter(x, y)
plt.savefig('scatter.png')
print('Scatter plot saved to scatter.png')
plt.show()
```

The code block is below: 
```javascript
const fs = require('fs');

// Define the content to write
const content = 'Hello';

// Write content to a file named 'output.txt'
fs.writeFile('from-js-output.txt', content, (err) => {
  if (err) {
    console.error('An error occurred while writing to the file:', err);
  } else {
    console.log('File has been written successfully!');
  }
});
```

The code block is below: 
```tsc
// Import the 'fs' module
import * as fs from 'fs';

// Define the content to write
const content: string = 'Hello';

// Write content to a file named 'output.txt'
fs.writeFile('from-ts-output.txt', content, (err: NodeJS.ErrnoException | null) => {
  if (err) {
    console.error('An error occurred while writing to the file:', err);
  } else {
    console.log('File has been written successfully!');
  }
});
```

This is the end of the message.

"""

# message_with_code_block="""
# This is a message with code block.
# 1.The code block is below:
# ```py
# print("hello docker!")
# ```
# the code block ends here
# """

# Generate a reply for the given code.
reply = code_executor_agent.generate_reply(messages=[{"role": "user", "content": message_with_code_block}])
print(reply)

print(os.listdir(temp_dir.name))