"""Command-line interface for PerturbGen."""


import click


class OrderedGroup(click.Group):
    """`click.Group` which prints its subcommands in a specific order.

    By default, Click will show subcommands in alphabetical order.
    Sometimes it makes more sense to use a different order, which you
    can manually specify by using this class.

    Usage:

        @click.group(
            cls=OrderedGroup,
            order=["foo", "bar"],
        )
        def main(): ...

        @main.command()
        def foo(): ...

        @main.command()
        def bar(): ...
    """
    def __init__(self, *args, order, **kwargs):
        super().__init__(*args, **kwargs)
        self.__order = order

    def list_commands(self, ctx):
        all_names = super().list_commands(ctx)
        all_names_set = set(all_names)
        ordered_names = [name for name in self.__order if name in all_names_set]
        other_names = [name for name in all_names if name not in self.__order]
        return ordered_names + other_names


@click.group(cls=OrderedGroup, order=["tokenise", "train-mask", "train-decoder", "generate"])
def main():
    pass


@main.command(context_settings={"ignore_unknown_options": True, "help_option_names": []})
@click.argument("args", nargs=-1)
def tokenise(args):
    """Data preprocessing, tokenisation"""
    click.echo("loading, please wait...")
    from perturbgen.pp.GF_tokenisation import main
    main(args)


@main.command(context_settings={"ignore_unknown_options": True, "help_option_names": []}, hidden=True)
@click.argument("args", nargs=-1)
def tokenize(args):
    return tokenise(args)


@main.command(context_settings={"ignore_unknown_options": True, "help_option_names": []})
@click.argument("args", nargs=-1)
def train_mask(args):
    """Training the masking model"""
    click.echo("loading, please wait...")
    from perturbgen.train import main
    main(args)


@main.command(context_settings={"ignore_unknown_options": True, "help_option_names": []})
@click.argument("args", nargs=-1)
def train_decoder(args):
    """Training the count decoder model"""
    click.echo("loading, please wait...")
    from perturbgen.train import main
    main(args)


@main.command(context_settings={"ignore_unknown_options": True, "help_option_names": []})
@click.argument("args", nargs=-1)
def generate(args):
    """Load checkpoint and generate predictions"""
    click.echo("loading, please wait...")
    from perturbgen.val import main
    main(args)


if __name__ == "__main__":
    main()
