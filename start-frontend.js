const { spawn } = require('child_process');
const path = require('path');

const frontendDir = path.join(__dirname, 'frontend');
process.chdir(frontendDir);

console.log('Starting TradingBot Frontend...');
console.log('Working directory:', process.cwd());

spawn('node', ['./node_modules/vite/bin/vite.js', '--host', '0.0.0.0', '--port', '5173'], {
  stdio: 'inherit',
  shell: false
});
