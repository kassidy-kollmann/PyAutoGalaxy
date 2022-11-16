import numpy as np
import pytest

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_2d_via_integral_from():

    gnfw = ag.mp.gNFWSph(
        centre=(0.0, 0.0), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0
    )

    deflections = gnfw.deflections_2d_via_integral_from(
        grid=np.array([[0.1875, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(0.43501, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.37701, 1e-3)

    gnfw = ag.mp.gNFWSph(
        centre=(0.3, 0.2), kappa_s=2.5, inner_slope=1.5, scale_radius=4.0
    )

    deflections = gnfw.deflections_2d_via_integral_from(
        grid=np.array([[0.1875, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(-9.31254, 1e-3)
    assert deflections[0, 1] == pytest.approx(-3.10418, 1e-3)

    gnfw = ag.mp.gNFW(
        centre=(0.0, 0.0),
        kappa_s=1.0,
        elliptical_comps=ag.convert.elliptical_comps_from(axis_ratio=0.3, angle=100.0),
        inner_slope=0.5,
        scale_radius=8.0,
    )
    deflections = gnfw.deflections_2d_via_integral_from(
        grid=np.array([[0.1875, 0.1625]])
    )
    assert deflections[0, 0] == pytest.approx(0.26604, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.58988, 1e-3)

    gnfw = ag.mp.gNFW(
        centre=(0.3, 0.2),
        kappa_s=2.5,
        elliptical_comps=ag.convert.elliptical_comps_from(axis_ratio=0.5, angle=100.0),
        inner_slope=1.5,
        scale_radius=4.0,
    )
    deflections = gnfw.deflections_2d_via_integral_from(
        grid=np.array([[0.1875, 0.1625]])
    )
    assert deflections[0, 0] == pytest.approx(-5.99032, 1e-3)
    assert deflections[0, 1] == pytest.approx(-4.02541, 1e-3)


def test__deflections_2d_via_mge_from():

    gnfw = ag.mp.gNFWSph(
        centre=(0.0, 0.0), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0
    )

    deflections_via_integral = gnfw.deflections_2d_via_integral_from(
        grid=np.array([[0.1875, 0.1625]])
    )
    deflections_via_mge = gnfw.deflections_2d_via_mge_from(
        grid=np.array([[0.1875, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_mge, 1.0e-3)

    gnfw = ag.mp.gNFWSph(
        centre=(0.3, 0.2), kappa_s=2.5, inner_slope=1.5, scale_radius=4.0
    )

    deflections_via_integral = gnfw.deflections_2d_via_integral_from(
        grid=np.array([[0.1875, 0.1625]])
    )
    deflections_via_mge = gnfw.deflections_2d_via_mge_from(
        grid=np.array([[0.1875, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_mge, 1.0e-3)

    gnfw = ag.mp.gNFW(
        centre=(0.0, 0.0),
        kappa_s=1.0,
        elliptical_comps=ag.convert.elliptical_comps_from(axis_ratio=0.3, angle=100.0),
        inner_slope=0.5,
        scale_radius=8.0,
    )

    deflections_via_integral = gnfw.deflections_2d_via_integral_from(
        grid=np.array([[0.1875, 0.1625]])
    )
    deflections_via_mge = gnfw.deflections_2d_via_mge_from(
        grid=np.array([[0.1875, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_mge, 1.0e-3)

    gnfw = ag.mp.gNFW(
        centre=(0.3, 0.2),
        kappa_s=2.5,
        elliptical_comps=ag.convert.elliptical_comps_from(axis_ratio=0.5, angle=100.0),
        inner_slope=1.5,
        scale_radius=4.0,
    )

    deflections_via_integral = gnfw.deflections_2d_via_integral_from(
        grid=np.array([[0.1875, 0.1625]])
    )
    deflections_via_mge = gnfw.deflections_2d_via_mge_from(
        grid=np.array([[0.1875, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_mge, 1.0e-3)


def test__deflections_yx_2d_from():

    gnfw = ag.mp.gNFW()

    deflections = gnfw.deflections_yx_2d_from(grid=np.array([[1.0, 0.0]]))
    deflections_via_mge = gnfw.deflections_2d_via_mge_from(grid=np.array([[1.0, 0.0]]))

    assert deflections == pytest.approx(deflections_via_mge, 1.0e-4)

    gnfw = ag.mp.gNFWSph()

    deflections = gnfw.deflections_yx_2d_from(grid=np.array([[1.0, 0.0]]))
    deflections_via_mge = gnfw.deflections_2d_via_mge_from(grid=np.array([[1.0, 0.0]]))

    assert deflections == pytest.approx(deflections_via_mge, 1.0e-4)

    elliptical = ag.mp.gNFW(
        centre=(0.1, 0.2),
        elliptical_comps=(0.0, 0.0),
        kappa_s=2.0,
        inner_slope=1.5,
        scale_radius=3.0,
    )
    spherical = ag.mp.gNFWSph(
        centre=(0.1, 0.2), kappa_s=2.0, inner_slope=1.5, scale_radius=3.0
    )

    assert elliptical.deflections_yx_2d_from(grid) == pytest.approx(
        spherical.deflections_yx_2d_from(grid), 1e-4
    )


def test__convergence_2d_via_mge_from():

    gnfw = ag.mp.gNFWSph(
        centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0
    )

    convergence = gnfw.convergence_2d_via_mge_from(grid=np.array([[2.0, 0.0]]))

    assert convergence == pytest.approx(0.30840, 1e-2)

    gnfw = ag.mp.gNFWSph(
        centre=(0.0, 0.0), kappa_s=2.0, inner_slope=1.5, scale_radius=1.0
    )

    convergence = gnfw.convergence_2d_via_mge_from(grid=np.array([[2.0, 0.0]]))

    assert convergence == pytest.approx(0.30840 * 2, 1e-2)

    gnfw = ag.mp.gNFW(
        centre=(0.0, 0.0),
        kappa_s=1.0,
        elliptical_comps=ag.convert.elliptical_comps_from(axis_ratio=0.5, angle=90.0),
        inner_slope=1.5,
        scale_radius=1.0,
    )
    assert gnfw.convergence_2d_via_mge_from(
        grid=np.array([[0.0, 1.0]])
    ) == pytest.approx(0.30840, 1e-2)

    gnfw = ag.mp.gNFW(
        centre=(0.0, 0.0),
        kappa_s=2.0,
        elliptical_comps=ag.convert.elliptical_comps_from(axis_ratio=0.5, angle=90.0),
        inner_slope=1.5,
        scale_radius=1.0,
    )
    assert gnfw.convergence_2d_via_mge_from(
        grid=np.array([[0.0, 1.0]])
    ) == pytest.approx(0.30840 * 2, 1e-2)


def test__convergence_2d_from():

    gnfw = ag.mp.gNFWSph(
        centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=1.0
    )

    convergence = gnfw.convergence_2d_from(grid=np.array([[2.0, 0.0]]))

    assert convergence == pytest.approx(0.30840, 1e-2)

    elliptical = ag.mp.gNFW(
        centre=(0.1, 0.2),
        elliptical_comps=(0.0, 0.0),
        kappa_s=2.0,
        inner_slope=1.5,
        scale_radius=3.0,
    )
    spherical = ag.mp.gNFWSph(
        centre=(0.1, 0.2), kappa_s=2.0, inner_slope=1.5, scale_radius=3.0
    )

    assert elliptical.potential_2d_from(grid) == pytest.approx(
        spherical.potential_2d_from(grid), 1e-4
    )


def test__potential_2d_from():

    gnfw = ag.mp.gNFWSph(
        centre=(0.0, 0.0), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0
    )

    potential = gnfw.potential_2d_from(grid=np.array([[0.1625, 0.1875]]))

    assert potential == pytest.approx(0.00920, 1e-3)

    gnfw = ag.mp.gNFWSph(
        centre=(0.0, 0.0), kappa_s=1.0, inner_slope=1.5, scale_radius=8.0
    )

    potential = gnfw.potential_2d_from(grid=np.array([[0.1625, 0.1875]]))

    assert potential == pytest.approx(0.17448, 1e-3)

    gnfw = ag.mp.gNFW(
        centre=(1.0, 1.0),
        kappa_s=5.0,
        elliptical_comps=ag.convert.elliptical_comps_from(axis_ratio=0.5, angle=100.0),
        inner_slope=1.0,
        scale_radius=10.0,
    )
    assert gnfw.potential_2d_from(grid=np.array([[2.0, 2.0]])) == pytest.approx(
        2.4718, 1e-4
    )

    elliptical = ag.mp.gNFW(
        centre=(0.1, 0.2),
        elliptical_comps=(0.0, 0.0),
        kappa_s=2.0,
        inner_slope=1.5,
        scale_radius=3.0,
    )
    spherical = ag.mp.gNFWSph(
        centre=(0.1, 0.2), kappa_s=2.0, inner_slope=1.5, scale_radius=3.0
    )

    assert elliptical.convergence_2d_from(grid) == pytest.approx(
        spherical.convergence_2d_from(grid), 1e-4
    )


def test__compare_to_nfw():

    nfw = ag.mp.NFW(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.111111),
        kappa_s=1.0,
        scale_radius=20.0,
    )
    gnfw = ag.mp.gNFW(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.111111),
        kappa_s=1.0,
        inner_slope=1.0,
        scale_radius=20.0,
    )

    assert nfw.deflections_yx_2d_from(grid) == pytest.approx(
        gnfw.deflections_yx_2d_from(grid), 1e-3
    )
    assert nfw.deflections_yx_2d_from(grid) == pytest.approx(
        gnfw.deflections_yx_2d_from(grid), 1e-3
    )

    assert nfw.potential_2d_from(grid) == pytest.approx(
        gnfw.potential_2d_from(grid), 1e-3
    )
    assert nfw.potential_2d_from(grid) == pytest.approx(
        gnfw.potential_2d_from(grid), 1e-3
    )
