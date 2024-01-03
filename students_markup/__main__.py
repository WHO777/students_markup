import argparse


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db-name",
        type=str,
        default="Students",
        help="postgres database (server) name",
    )
    parser.add_argument(
        "--db-user",
        type=str,
        default="postgres",
        help="postgres user name used to authenticate",
    )
    parser.add_argument(
        "--db-password",
        type=str,
        default="postgres",
        help="postgres user password used to authenticate",
    )
    parser.add_argument(
        "--db-host",
        type=str,
        default="localhost",
        help="database host address",
    )
    parser.add_argument(
        "--db-port", type=int, default="5432", help="connection port number"
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    config = parse_config()
    print(config)
