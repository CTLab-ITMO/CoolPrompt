import minio
import urllib3
import argparse


urllib3.disable_warnings()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ip", default="94.124.179.195")
    parser.add_argument("--port", default="9000")
    parser.add_argument("--login")
    parser.add_argument("--password")
    parser.add_argument("--save_path")

    return parser.parse_args()


def main():
    args = parse_args()

    client = minio.Minio(
        f"{args.ip}:{args.port}",
        access_key=args.login,
        secret_key=args.password,
        cert_check=False
    )

    bucket_name = 'autoprompting'

    client.fget_object(
        bucket_name,
        'data/autoprompting_datasets.zip',
        args.save_path
    )


if __name__ == "__main__":
    main()
