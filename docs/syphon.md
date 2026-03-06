# Using Syphon

Scope supports near-zero latency sharing of real-time video with other local applications on macOS via [Syphon](https://syphon.info/).

> [!IMPORTANT]
> Syphon requires macOS 11 or later. It is not available on Windows or Linux.

## Syphon Receiver

Scope can receive video from any Syphon server running on the same Mac:

1. Select "Video" for "Input Mode" under "Input & Controls".
2. Select "Syphon" for "Video Source".
3. Click the refresh button to discover available Syphon servers.
4. Select the server you want to receive from in the dropdown. A live preview is shown automatically.

## Compatible Applications

Scope should be able to receive video from any application that publishes a Syphon server.

Common Syphon-compatible applications include:

- [TouchDesigner](https://derivative.ca/) (with Syphon Spout Out TOP)
- [Resolume](https://resolume.com/)
- [MadMapper](https://madmapper.com/)
- [VDMX](https://vidvox.net/)
- [OBS](https://obsproject.com/) (with Syphon plugin)
