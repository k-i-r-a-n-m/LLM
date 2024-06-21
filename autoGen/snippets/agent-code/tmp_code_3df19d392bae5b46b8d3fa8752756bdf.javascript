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