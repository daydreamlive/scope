import type { ForgeConfig } from '@electron-forge/shared-types';
import { MakerSquirrel } from '@electron-forge/maker-squirrel';
import { MakerZIP } from '@electron-forge/maker-zip';
import { MakerDeb } from '@electron-forge/maker-deb';
import { MakerRpm } from '@electron-forge/maker-rpm';
import { VitePlugin } from '@electron-forge/plugin-vite';
import { FusesPlugin } from '@electron-forge/plugin-fuses';
import { FuseV1Options, FuseVersion } from '@electron/fuses';
import { MakerDMG } from '@electron-forge/maker-dmg';
import path from 'path';

const config: ForgeConfig = {
  packagerConfig: {
    name: "DaydreamScope",
    executableName: "daydream-scope",
    icon: path.resolve(__dirname, "assets", "icon"),
    asar: false, // We need to access Python files, so disable asar
    // Note: Python project files need to be copied to resourcesPath during packaging
    // This is typically done via hooks or by including them in the app directory
  },
  hooks: {
    async packageAfterCopy(_forgeConfig, buildPath) {
      // Copy Python project files to the build directory
      const fs = await import('fs/promises');
      const projectRoot = path.resolve(__dirname, '..');
      const destRoot = path.join(buildPath, '..', 'resources', 'app');

      // Ensure destination exists
      await fs.mkdir(destRoot, { recursive: true });

      // Copy necessary files
      const filesToCopy = [
        { src: path.join(projectRoot, 'src'), dest: path.join(destRoot, 'src') },
        { src: path.join(projectRoot, 'pyproject.toml'), dest: path.join(destRoot, 'pyproject.toml') },
        { src: path.join(projectRoot, 'uv.lock'), dest: path.join(destRoot, 'uv.lock') },
        { src: path.join(projectRoot, 'README.md'), dest: path.join(destRoot, 'README.md') },
        { src: path.join(projectRoot, 'LICENSE.md'), dest: path.join(destRoot, 'LICENSE.md') },
        { src: path.join(projectRoot, 'frontend', 'dist'), dest: path.join(destRoot, 'frontend', 'dist') },
        { src: path.join(__dirname, 'assets'), dest: path.join(destRoot, 'assets') },
      ];

      for (const { src, dest } of filesToCopy) {
        try {
          const stat = await fs.stat(src);
          if (stat.isDirectory()) {
            await fs.cp(src, dest, { recursive: true });
          } else {
            await fs.mkdir(path.dirname(dest), { recursive: true });
            await fs.copyFile(src, dest);
          }
        } catch (err) {
          console.warn(`Failed to copy ${src} to ${dest}:`, err);
        }
      }
    },
  },
  makers: [
    new MakerSquirrel({
      name: "daydream-scope",
    }),
    new MakerZIP({}, ['darwin']),
    new MakerRpm({
      options: {
        name: "daydream-scope",
      },
    }),
    new MakerDeb({
      options: {
        name: "daydream-scope",
      },
    }),
    new MakerDMG({
      name: "DaydreamScope",
    }),
  ],
  plugins: [
    new VitePlugin({
      build: [
        {
          entry: 'src/main.ts',
          config: 'vite.main.config.ts',
        },
        {
          entry: 'src/preload.ts',
          config: 'vite.preload.config.ts',
        },
      ],
      renderer: [
        {
          name: 'main_window',
          config: 'vite.renderer.config.ts',
        },
      ],
    }),
    new FusesPlugin({
      version: FuseVersion.V1,
      [FuseV1Options.RunAsNode]: false,
      [FuseV1Options.EnableCookieEncryption]: false,
      [FuseV1Options.EnableNodeOptionsEnvironmentVariable]: false,
      [FuseV1Options.EnableNodeCliInspectArguments]: false,
      [FuseV1Options.EnableEmbeddedAsarIntegrityValidation]: true,
      [FuseV1Options.OnlyLoadAppFromAsar]: false,
    }),
  ],
};

export default config;
