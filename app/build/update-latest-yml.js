#!/usr/bin/env node
/**
 * Update latest.yml / latest-mac.yml after code signing
 *
 * Usage:
 *   node update-latest-yml.js                  # Windows (default)
 *   node update-latest-yml.js --platform win   # Windows (explicit)
 *   node update-latest-yml.js --platform mac   # macOS
 *
 * When code signing happens outside of electron-builder (e.g., Azure Trusted Signing
 * on Windows, or Apple notarization on macOS), the binary is modified and the SHA512
 * checksum in latest.yml becomes invalid.  This script regenerates the yml with the
 * correct checksums.
 */

const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const DIST_DIR = path.join(__dirname, '..', 'dist');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function calculateSha512(filePath) {
  const fileBuffer = fs.readFileSync(filePath);
  const hashSum = crypto.createHash('sha512');
  hashSum.update(fileBuffer);
  return hashSum.digest('base64');
}

function parsePlatformArg() {
  const idx = process.argv.indexOf('--platform');
  if (idx === -1 || idx + 1 >= process.argv.length) {
    return 'win'; // default for backward compatibility
  }
  return process.argv[idx + 1];
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

function findExeFile() {
  const files = fs.readdirSync(DIST_DIR);
  const exeFile = files.find(f => f.endsWith('.exe') && !f.includes('Uninstall'));
  if (!exeFile) {
    throw new Error('No exe file found in dist directory');
  }
  return path.join(DIST_DIR, exeFile);
}

function updateLatestYmlWin(exeFilePath) {
  console.log('Updating latest.yml with signed exe checksum...');

  const packageJson = require('../package.json');
  const version = packageJson.version;
  const exeFileName = path.basename(exeFilePath);

  const exeSha512 = calculateSha512(exeFilePath);
  const exeSize = fs.statSync(exeFilePath).size;

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

// ---------------------------------------------------------------------------
// macOS
// ---------------------------------------------------------------------------

function findZipFile(arch) {
  const files = fs.readdirSync(DIST_DIR);
  // Match zip files with the architecture in the name
  const zipFile = files.find(f => f.endsWith('.zip') && f.includes(arch));
  if (!zipFile) {
    throw new Error(`ZIP not found for arch ${arch} in ${DIST_DIR}`);
  }
  return path.join(DIST_DIR, zipFile);
}

function updateLatestYmlMac() {
  console.log('Generating latest-mac.yml ...');

  const packageJson = require('../package.json');
  const version = packageJson.version;
  const releaseDate = new Date().toISOString();

  const arm64Path = findZipFile('arm64');
  const x64Path = findZipFile('x64');

  const arm64Name = path.basename(arm64Path);
  const x64Name = path.basename(x64Path);

  const arm64Sha512 = calculateSha512(arm64Path);
  const x64Sha512 = calculateSha512(x64Path);

  const arm64Size = fs.statSync(arm64Path).size;
  const x64Size = fs.statSync(x64Path).size;

  const latestMacYml = `version: ${version}
files:
  - url: ${arm64Name}
    sha512: ${arm64Sha512}
    size: ${arm64Size}
    arch: arm64
  - url: ${x64Name}
    sha512: ${x64Sha512}
    size: ${x64Size}
    arch: x64
path: ${arm64Name}
sha512: ${arm64Sha512}
releaseDate: ${releaseDate}
`;

  const ymlPath = path.join(DIST_DIR, 'latest-mac.yml');
  fs.writeFileSync(ymlPath, latestMacYml, 'utf8');
  console.log(`✓ latest-mac.yml updated: ${ymlPath}`);
  console.log('\nContents:');
  console.log(latestMacYml);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

function main() {
  const platform = parsePlatformArg();

  try {
    if (platform === 'mac') {
      console.log('=== Generating latest-mac.yml ===\n');
      updateLatestYmlMac();
      console.log('\n✓ latest-mac.yml generated successfully!');
    } else {
      console.log('=== Updating latest.yml after code signing ===\n');
      const exeFilePath = findExeFile();
      console.log(`Found exe file: ${exeFilePath}\n`);
      updateLatestYmlWin(exeFilePath);
      console.log('\n✓ latest.yml updated successfully!');
    }
  } catch (error) {
    console.error('\n✗ Error:', error.message);
    process.exit(1);
  }
}

main();
