const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);

async function killPort(port) {
  try {
    if (process.platform === 'win32') {
      // Windows
      const { stdout } = await execPromise(`netstat -ano | findstr :${port}`);
      const lines = stdout.trim().split('\n');
      
      for (const line of lines) {
        const parts = line.trim().split(/\s+/);
        const pid = parts[parts.length - 1];
        
        if (pid && !isNaN(pid)) {
          try {
            await execPromise(`taskkill /F /PID ${pid}`);
            console.log(`✓ Processo sulla porta ${port} terminato (PID: ${pid})`);
          } catch (err) {
            // Ignora errori se il processo è già terminato
          }
        }
      }
    } else {
      // Unix-like systems
      try {
        await execPromise(`lsof -ti:${port} | xargs kill -9`);
        console.log(`✓ Processo sulla porta ${port} terminato`);
      } catch (err) {
        // Porta già libera
      }
    }
  } catch (error) {
    // Porta già libera o nessun processo trovato
    console.log(`✓ Porta ${port} è libera`);
  }
}

const port = process.argv[2] || 3000;
killPort(port).then(() => {
  console.log(`Pronto per avviare sulla porta ${port}`);
  process.exit(0);
}).catch(err => {
  console.error('Errore:', err.message);
  process.exit(0); // Continua comunque
});
