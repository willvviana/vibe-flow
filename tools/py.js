// Run the project's virtualenv Python on any OS:  node tools/py.js -m pytest tests -q
const { spawnSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const py = path.join(__dirname, '..', '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
if (!fs.existsSync(py)) { console.error('Python environment missing. Run: npm run setup'); process.exit(1); }
process.exit(spawnSync(py, process.argv.slice(2), { stdio: 'inherit', cwd: path.join(__dirname, '..') }).status ?? 1);
