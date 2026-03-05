from pythonosc.udp_client import SimpleUDPClient

HOST = "127.0.0.1"
PORT = 8000
OSC_ADDRESS = "/scope/text"


def main():
    client = SimpleUDPClient(HOST, PORT)
    print(f"Connected to OSC at {HOST}:{PORT}")
    print("Enter prompts to send. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            prompt = input("Prompt> ").strip()

            if prompt.lower() in ["exit", "quit"]:
                print("Goodbye.")
                break

            if not prompt:
                continue

            client.send_message(OSC_ADDRESS, prompt)
            print(f"Sent: {prompt}")

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting.")
            break


if __name__ == "__main__":
    main()
