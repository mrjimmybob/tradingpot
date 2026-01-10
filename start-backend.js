const { spawn } = require('child_process');
const path = require('path');

const backendDir = path.join(__dirname, 'backend');

// Try different Python commands
const pythonCommands = ['python', 'python3', 'py'];

async function tryCommand(cmd, args, cwd) {
  return new Promise((resolve) => {
    const proc = spawn(cmd, args, { cwd, stdio: 'inherit', shell: true });
    proc.on('error', () => resolve(false));
    proc.on('spawn', () => resolve(proc));
  });
}

async function main() {
  console.log('Starting TradingBot Backend...');

  // First, try to install requirements
  for (const py of pythonCommands) {
    console.log(`Trying ${py}...`);
    const result = await tryCommand(py, ['-m', 'pip', 'install', '-r', 'requirements.txt'], backendDir);
    if (result) {
      console.log(`Using ${py}`);
      // Now start uvicorn
      spawn(py, ['-m', 'uvicorn', 'app.main:app', '--reload', '--host', '0.0.0.0', '--port', '8000'], {
        cwd: backendDir,
        stdio: 'inherit',
        shell: true
      });
      return;
    }
  }

  console.error('No Python interpreter found!');
  process.exit(1);
}

main();
