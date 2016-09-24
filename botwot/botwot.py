import os
import time
import aiml
from slackclient import SlackClient


# starterbot's ID as an environment variable
BOT_ID = str(os.environ.get("BOT_ID"))

# constants
AT_BOT = "<@"+BOT_ID+">" 
EXAMPLE_COMMAND = "do"

# instantiate Slack & Twilio clients
slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))

# Instantiate AIML kernel and learn the AIML files.
kernel = aiml.Kernel()
if os.path.isfile("botwot_brain.brn"):
    kernel.bootstrap(brainFile="botwot_brain.brn")
else:
    kernel.bootstrap(learnFiles="std-startup.xml", commands="lol")
    kernel.saveBrain("botwot_brain.brn")


def handle_command(command, channel):
    """
    Returns back responses based on the users input
    :param command: Contains the user input.
    :param channel: Channel to which the result should be published
    :return: None
    """
    response = kernel.respond(command)
    slack_client.api_call("chat.postMessage", channel=channel,
                          text=response, as_user=True)


def parse_slack_output(slack_rtm_output):
    """
    Function to parse the RTM API output and fetch messages directed at the bot
    :param slack_rtm_output: Slack RTM API output event firehose
    :return: command:- User text for the bot
             channel:- Channel through which the event was received.
    """
    command = None
    channel = None
    output_list = slack_rtm_output
    if output_list and len(output_list) > 0:
        for output in output_list:
            if output and 'text' in output and AT_BOT in output['text']:
                # return text after the @ mention, whitespace removed
                command = output['text'].split(AT_BOT)[1].strip().lower()
                channel = output['channel']
                return command, channel

    return command, channel


if __name__ == "__main__":
    READ_WEBSOCKET_DELAY = 1  # 1 second delay between reading from firehose
    if slack_client.rtm_connect():
        print("Botwot connected and running!")
        while True:
            command, channel = parse_slack_output(slack_client.rtm_read())
            if command and channel:
                handle_command(command, channel)
            time.sleep(READ_WEBSOCKET_DELAY)
    else:
        print("Connection failed. Invalid Slack token or bot ID?")
