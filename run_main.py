import click
from splatviz import Splatviz
from stereo_window import StereoWindow


@click.command()
@click.option("--gan_path", help="path to GAN project", required=True, type=click.Path())
def main(gan_path):
    splatviz = Splatviz(gan_path=gan_path)
    while not splatviz.should_close():
        splatviz.draw_frame()
    splatviz.close()


if __name__ == "__main__":
    main()
