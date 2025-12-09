#!/usr/bin/env node
/**
 * Update latest.yml after code signing
 *
 * When code signing happens outside of electron-builder (e.g., Azure Trusted Signing),
 * the binary is modified and the SHA512 checksum in latest.yml becomes invalid.
 * This script regenerates latest.yml with the correct checksum for the signed exe.
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const DIST_DIR = path.join(__dirname, '..', 'dist');

function calculateSha512(filePath) {
  const fileBuffer = fs.readFileSync(filePath);
  const hashSum = crypto.createHash('sha512');
  hashSum.update(fileBuffer);
  return hashSum.digest('base64');
}

function findExeFile() {
  const files = fs.readdirSync(DIST_DIR);
  const exeFile = files.find(f => f.endsWith('.exe') && !f.includes('Uninstall'));
  if (!exeFile) {
    throw new Error('No exe file found in dist directory');
  }
  return path.join(DIST_DIR, exeFile);
}

function updateLatestYml(exeFilePath) {
  console.log('Updating latest.yml with signed exe checksum...');

  const packageJson = require('../package.json');
  const version = packageJson.version;
  const exeFileName = path.basename(exeFilePath);

  // Calculate checksum of signed exe
  const exeSha512 = calculateSha512(exeFilePath);
  const exeSize = fs.statSync(exeFilePath).size;

  // Generate latest.yml in the same format electron-builder produces
  // Note: We don't include blockMapSize since we're not using differential updates
  const latestYml = `version: ${version}
files:
  - url: ${exeFileName}
    sha512: ${exeSha512}
    size: ${exeSize}
path: ${exeFileName}
sha512: ${exeSha512}
releaseDate: ${new Date().toISOString()}
`;

  const latestYmlPath = path.join(DIST_DIR, 'latest.yml');
  fs.writeFileSync(latestYmlPath, latestYml, 'utf8');
  console.log(`✓ latest.yml updated: ${latestYmlPath}`);
  console.log('\nContents:');
  console.log(latestYml);
}

function main() {
  try {
    console.log('=== Updating latest.yml after code signing ===\n');

    const exeFilePath = findExeFile();
    console.log(`Found exe file: ${exeFilePath}\n`);

    updateLatestYml(exeFilePath);

    console.log('\n✓ latest.yml updated successfully!');
  } catch (error) {
    console.error('\n✗ Error:', error.message);
    process.exit(1);
  }
}

main();
