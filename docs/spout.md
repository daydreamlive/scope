# Using Spout

Scope supports near-zero latency and overhead sharing of real-time video with other local applications on Windows via [Spout](https://spout.zeal.co/).

> [!IMPORTANT]
> The Scope server *must* be running on a Windows machine to use Spout. This means that Spout is not available if you run Scope on a Runpod (Linux) machine.

## Spout Receiver

Scope can be configured to use video from Spout by configuring the Spout Receiver:

1. Select "Video" for "Input Mode" under "Input & Controls".
2. Select "Spout Receiver" for "Video Source".
3. The "Sender Name" can be left empty to receive video from any application or it can be set to the specific name you have set in the sending application.

<img width="506" height="650" alt="Screenshot 2025-12-15 133937" src="https://github.com/user-attachments/assets/b513b2e4-1cfe-407f-955b-07635caf044d" />

## Spout Sender

Scope can be configured to send video with Spout by configuring the Spout Sender:

1. Toggle "Spout Sender" to ON under "Settings".
2. The video will be sent with the "Sender Name" set to the default of "ScopeOut", but you can change this as well.

<img width="506" height="1052" alt="Screenshot 2025-12-15 134013" src="https://github.com/user-attachments/assets/fe0946ae-35fc-439f-84bb-e9ae0444757c" />

## Compatible Applications

Scope should be able to share real-time video with any application that supports Spout.

The following applications have been tested thus far:

- [TouchDesigner](https://derivative.ca/) (with [Syphon Spout In](https://derivative.ca/UserGuide/Syphon_Spout_In_TOP) and [Syphon Spout Out](https://docs.derivative.ca/Syphon_Spout_Out_TOP) TOPs)
- [Unity](https://unity.com/) (with [KlakSpout plugin](https://github.com/keijiro/KlakSpout))
- [Blender](https://www.blender.org/) (with [TextureSharing add-on](https://github.com/maybites/TextureSharing))

## Example Usage (TouchDesigner)

https://github.com/user-attachments/assets/62a197fb-1408-466c-b272-783cf66edfef
