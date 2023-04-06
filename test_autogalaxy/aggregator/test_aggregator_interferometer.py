from os import path

from autoconf import conf
import autofit as af
import autogalaxy as ag

from test_autogalaxy.aggregator.conftest import clean


def test__interferometer_generator_from_aggregator(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
    samples,
    model,
):

    path_prefix = "aggregator_interferometer"

    database_file = path.join(conf.instance.output_path, "interferometer.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    interferometer_7 = ag.Interferometer(
        data=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        settings=ag.SettingsInterferometer(
            grid_class=ag.Grid2DIterate,
            grid_pixelization_class=ag.Grid2DIterate,
            fractional_accuracy=0.5,
            sub_steps=[2],
            transformer_class=ag.TransformerDFT,
        ),
    )

    search = ag.m.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    interferometer_agg = ag.agg.InterferometerAgg(aggregator=agg)
    interferometer_gen = interferometer_agg.interferometer_gen_from()

    for interferometer in interferometer_gen:
        assert (interferometer.visibilities == interferometer_7.visibilities).all()
        assert (interferometer.real_space_mask == mask_2d_7x7).all()
        assert isinstance(interferometer.grid, ag.Grid2DIterate)
        assert isinstance(interferometer.grid_pixelization, ag.Grid2DIterate)
        assert interferometer.grid.sub_steps == [2]
        assert interferometer.grid.fractional_accuracy == 0.5
        assert isinstance(interferometer.transformer, ag.TransformerDFT)

    clean(database_file=database_file, result_path=result_path)
