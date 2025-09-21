import click
from splatviz import Splatviz


@click.command()
@click.option("--gan_path", help="path to GAN project", default="./cgs_gan", type=click.Path())
def main(gan_path):
    splatviz = Splatviz(gan_path=gan_path)
    while not splatviz.should_close():
        splatviz.draw_frame()
    splatviz.close()


if __name__ == "__main__":
    main()
