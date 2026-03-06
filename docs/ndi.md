# Using NDI

Scope supports real-time, low-latency video over IP via [NDI (Network Device Interface)](https://ndi.video/). Unlike [Spout](spout.md) (Windows) and [Syphon](syphon.md) (macOS) which are limited to sharing between applications on the same machine, NDI works across the local network — you can send and receive video between different computers.

> [!IMPORTANT]
> The [NDI SDK / NDI Tools](https://ndi.video/tools/) must be installed on the machine running Scope. NDI works on Windows, macOS, and Linux.

## NDI Receiver

Scope can receive video from any NDI source on the network:

1. Select "Video" for "Input Mode" under "Input & Controls".
2. Select "NDI" for "Video Source".
3. Available NDI sources on the network will appear in the dropdown. Select the one you want to receive from.
4. A small preview thumbnail will show the current frame from the selected source.

> [!NOTE]
> When selecting an NDI source, Scope automatically probes the source's native resolution and adjusts the pipeline dimensions to match. This avoids stretching or compression artifacts.

## NDI Sender

> [!NOTE]
> NDI output (sender) support is coming soon. Once available, you will be able to send Scope's processed output over NDI to any receiver on the network.

## Installing NDI Tools

1. Go to [https://ndi.video/tools/](https://ndi.video/tools/) and download the installer for your platform:
   - **Windows**: Install "NDI Tools" (includes the runtime DLL).
   - **macOS**: Install "NDI SDK for Apple" (provides `libndi.dylib`).
   - **Linux**: Install the NDI SDK and ensure `libndi.so` is on the library path.
2. Restart Scope after installation. The NDI option will appear automatically once the SDK is detected. If it doesn't appear, try restarting Scope again. On some occasions (particularly on Windows if environment variables aren't being picked up), you may need to restart your machine.

## Compatible Applications

Scope should be able to share real-time video with any application that supports NDI.

The following applications have been tested thus far:

- [Resolume Arena / Avenue](https://resolume.com/) (built-in NDI input/output)
- [OBS Studio](https://obsproject.com/) (with [DistroAV plugin](https://github.com/DistroAV/DistroAV))
- [TouchDesigner](https://derivative.ca/) (with [NDI In](https://docs.derivative.ca/NDI_In_TOP) and [NDI Out](https://docs.derivative.ca/NDI_Out_TOP) TOPs)
- [vMix](https://www.vmix.com/) (built-in NDI support)
- [VirtualDJ](https://ndi.video/tools/) (with Send-NDI plugin)
