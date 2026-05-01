/**
 * afterSign hook for electron-builder — Apple notarization
 *
 * Submits the .app bundle to Apple for notarization and staples the result.
 * Skips gracefully when not on macOS or when credentials are missing
 * (allows local dev builds and unsigned CI builds without an Apple Developer account).
 *
 * Expected env variables (matching daydream-obs codesigning action):
 *   MACOS_SIGNING_IDENTITY       — e.g. "Developer ID Application: Company (TEAMID)"
 *   MACOS_NOTARIZATION_USERNAME  — Apple ID email
 *   MACOS_NOTARIZATION_PASSWORD  — App-specific password
 */
const { notarize } = require('@electron/notarize');

exports.default = async function notarizing(context) {
  const { electronPlatformName, appOutDir } = context;

  if (electronPlatformName !== 'darwin') {
    console.log('Skipping notarization — not macOS.');
    return;
  }

  if (!process.env.MACOS_NOTARIZATION_USERNAME || !process.env.MACOS_NOTARIZATION_PASSWORD) {
    console.log('Skipping notarization — MACOS_NOTARIZATION_USERNAME or MACOS_NOTARIZATION_PASSWORD not set.');
    return;
  }

  // Extract team ID from signing identity, e.g. "Developer ID Application: Company (ABC123)" → "ABC123"
  const identity = process.env.MACOS_SIGNING_IDENTITY || '';
  const teamIdMatch = identity.match(/\(([A-Z0-9]+)\)\s*$/);
  if (!teamIdMatch) {
    console.log('Skipping notarization — could not extract team ID from MACOS_SIGNING_IDENTITY.');
    return;
  }
  const teamId = teamIdMatch[1];

  const appName = context.packager.appInfo.productFilename;
  const appPath = `${appOutDir}/${appName}.app`;

  console.log(`Notarizing ${appPath} (team ${teamId}) ...`);

  await notarize({
    appPath,
    appleId: process.env.MACOS_NOTARIZATION_USERNAME,
    appleIdPassword: process.env.MACOS_NOTARIZATION_PASSWORD,
    teamId,
  });

  console.log('Notarization complete.');
};
