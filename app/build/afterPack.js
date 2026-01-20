/**
 * afterPack hook for electron-builder
 * This copies Python project files to the build directory after packaging
 */
const fs = require('fs-extra');
const path = require('path');

exports.default = async function afterPack(context) {
  const { electronPlatformName, appOutDir } = context;
  console.log(`Running afterPack hook for ${electronPlatformName}...`);

  const projectRoot = path.resolve(__dirname, '..', '..');
  let resourcesPath;

  // Determine resources path based on platform
  if (electronPlatformName === 'darwin') {
    // Find the .app bundle dynamically (handles different app names and architectures)
    // This is more reliable than hardcoding the app name
    const entries = fs.readdirSync(appOutDir);
    const appBundle = entries.find(entry => entry.endsWith('.app'));
    if (!appBundle) {
      throw new Error(`Could not find .app bundle in ${appOutDir}`);
    }
    resourcesPath = path.join(appOutDir, appBundle, 'Contents', 'Resources');
    console.log(`Found app bundle: ${appBundle}`);
  } else if (electronPlatformName === 'win32') {
    resourcesPath = path.join(appOutDir, 'resources');
  } else {
    // Linux
    resourcesPath = path.join(appOutDir, 'resources');
  }

  console.log(`Resources path: ${resourcesPath}`);

  // Files and directories to copy
  const filesToCopy = [
    { src: path.join(projectRoot, 'src'), dest: path.join(resourcesPath, 'src') },
    { src: path.join(projectRoot, 'pyproject.toml'), dest: path.join(resourcesPath, 'pyproject.toml') },
    { src: path.join(projectRoot, 'uv.lock'), dest: path.join(resourcesPath, 'uv.lock') },
    { src: path.join(projectRoot, '.python-version'), dest: path.join(resourcesPath, '.python-version') },
    { src: path.join(projectRoot, 'README.md'), dest: path.join(resourcesPath, 'README.md') },
    { src: path.join(projectRoot, 'LICENSE.md'), dest: path.join(resourcesPath, 'LICENSE.md') },
    { src: path.join(projectRoot, 'frontend', 'dist'), dest: path.join(resourcesPath, 'frontend', 'dist') },
  ];

  // Copy files
  for (const { src, dest } of filesToCopy) {
    try {
      const exists = await fs.pathExists(src);
      if (exists) {
        await fs.copy(src, dest, { overwrite: true });
        console.log(`✓ Copied: ${src} → ${dest}`);
      } else {
        console.warn(`⚠ Source not found: ${src}`);
      }
    } catch (err) {
      console.error(`✗ Failed to copy ${src} to ${dest}:`, err.message);
    }
  }

  console.log('afterPack hook completed.');
};
