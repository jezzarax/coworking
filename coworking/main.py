import logging

import click
from dotenv import load_dotenv

from coworking import chatbox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [line:%(lineno)d] %(message)s",
)

load_dotenv()


@click.group
def main_grp():
    print("Hello from coworking!")


main_grp.add_command(chatbox.chatbox_sim)

if __name__ == "__main__":
    main_grp()
