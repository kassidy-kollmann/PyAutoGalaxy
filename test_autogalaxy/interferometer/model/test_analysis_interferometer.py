from os import path

import autofit as af
import autogalaxy as ag

from autogalaxy.interferometer.model.result import ResultInterferometer


directory = path.dirname(path.realpath(__file__))


def test__make_result__result_interferometer_is_returned(interferometer_7):
    model = af.Collection(galaxies=af.Collection(galaxy_0=ag.Galaxy(redshift=0.5)))

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

    def modify_after_fit(
        paths: af.DirectoryPaths, model: af.AbstractPriorModel, result: af.Result
    ):
        pass

    analysis.modify_after_fit = modify_after_fit

    search = ag.m.MockSearch(name="test_search")

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultInterferometer)


def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
    interferometer_7,
):
    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

    instance = model.instance_from_unit_vector([])
    fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

    galaxies = analysis.galaxies_via_instance_from(instance=instance)

    fit = ag.FitInterferometer(dataset=interferometer_7, galaxies=galaxies)

    assert fit.log_likelihood == fit_figure_of_merit


def test__profile_log_likelihood_function(interferometer_7):
    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    galaxy = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    instance = model.instance_from_unit_vector([])

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

    run_time_dict, info_dict = analysis.profile_log_likelihood_function(
        instance=instance
    )

    assert "regularization_term_0" in run_time_dict
    assert "log_det_regularization_matrix_term_0" in run_time_dict
