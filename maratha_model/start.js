#!/usr/bin/env node
const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const rootDir = __dirname;
const backendDir = path.join(rootDir, 'backend');
const frontendDir = path.join(rootDir, 'frontend');

// Find Python/Uvicorn binary in backend .venv or system
let uvicornCmd = 'uvicorn';
let uvicornArgs = ['app.main:app', '--port', '8005', '--reload'];

const venvUvicorn = path.join(backendDir, '.venv', 'bin', 'uvicorn');
const venvPython = path.join(backendDir, '.venv', 'bin', 'python');

if (fs.existsSync(venvUvicorn)) {
  uvicornCmd = venvUvicorn;
} else if (fs.existsSync(venvPython)) {
  uvicornCmd = venvPython;
  uvicornArgs = ['-m', 'uvicorn', 'app.main:app', '--port', '8005', '--reload'];
}

console.log('\x1b[36m%s\x1b[0m', '═══════════════════════════════════════════════════════════');
console.log('\x1b[1m\x1b[32m%s\x1b[0m', '  🚀 Starting InfraPulse Full-Stack System (Backend + Frontend)');
console.log('\x1b[36m%s\x1b[0m', '═══════════════════════════════════════════════════════════\n');

// 1. Start Backend
const backend = spawn(uvicornCmd, uvicornArgs, {
  cwd: backendDir,
  shell: true,
  stdio: 'pipe',
  env: { ...process.env, PYTHONUNBUFFERED: '1' }
});

backend.stdout.on('data', (data) => {
  process.stdout.write(`\x1b[34m[backend]\x1b[0m ${data}`);
});
backend.stderr.on('data', (data) => {
  process.stderr.write(`\x1b[34m[backend]\x1b[0m ${data}`);
});

// 2. Start Frontend
const npmCmd = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const frontend = spawn(npmCmd, ['run', 'dev'], {
  cwd: frontendDir,
  shell: true,
  stdio: 'pipe'
});

frontend.stdout.on('data', (data) => {
  process.stdout.write(`\x1b[32m[frontend]\x1b[0m ${data}`);
});
frontend.stderr.on('data', (data) => {
  process.stderr.write(`\x1b[32m[frontend]\x1b[0m ${data}`);
});

// Graceful exit handling
function cleanExit() {
  console.log('\n\x1b[33m%s\x1b[0m', 'Shutting down InfraPulse backend and frontend...');
  try { backend.kill('SIGTERM'); } catch {}
  try { frontend.kill('SIGTERM'); } catch {}
  process.exit(0);
}

process.on('SIGINT', cleanExit);
process.on('SIGTERM', cleanExit);
