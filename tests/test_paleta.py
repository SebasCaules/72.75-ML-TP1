"""Correr con: python -m tests.test_paleta"""

import numpy as np

from src.graficos import (
    COLOR_CATEGORICA,
    COLOR_FUMADOR_CLARO,
    COLOR_FUMADOR_OSCURO,
    COLOR_NUMERICA,
    COLOR_OBJETIVO,
    SUPERFICIE,
)

PISO_DELTA_E = 8.0
PISO_CONTRASTE = 3.0
PISO_CROMA = 5.0

MATRICES_DALTONISMO = {
    "protanopía": np.array([[0.1121, 0.8853, -0.0005],
                            [0.1127, 0.8897, -0.0001],
                            [0.0045, 0.0000, 1.0019]]),
    "deuteranopía": np.array([[0.2920, 0.7054, -0.0003],
                              [0.2934, 0.7089, 0.0000],
                              [-0.0202, 0.0270, 0.9915]]),
    "tritanopía": np.array([[1.0170, 0.1097, -0.1269],
                            [0.0000, 0.9578, 0.0423],
                            [0.0000, 0.3355, 0.6645]]),
}

_A_LMS = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
_A_LAB = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])


def _a_lineal(color_hex):
    h = color_hex.lstrip("#")
    canales = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255
    return np.where(canales <= 0.04045, canales / 12.92, ((canales + 0.055) / 1.055) ** 2.4)


def _a_oklab(lineal):
    return _A_LAB @ np.cbrt(np.clip(_A_LMS @ lineal, 0, None))


def luminancia(color_hex):
    return float(np.dot([0.2126, 0.7152, 0.0722], _a_lineal(color_hex)))


def contraste(color_a, color_b):
    la, lb = luminancia(color_a), luminancia(color_b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def croma(color_hex):
    _, a, b = _a_oklab(_a_lineal(color_hex))
    return float(np.hypot(a, b) * 100)


def delta_e(color_a, color_b, daltonismo=None):
    lin_a, lin_b = _a_lineal(color_a), _a_lineal(color_b)
    if daltonismo is not None:
        m = MATRICES_DALTONISMO[daltonismo]
        lin_a, lin_b = m @ lin_a, m @ lin_b
    return float(np.linalg.norm(_a_oklab(lin_a) - _a_oklab(lin_b)) * 100)


def test_negro_y_blanco_dan_el_contraste_maximo():
    assert abs(contraste("#000000", "#ffffff") - 21.0) < 1e-9
    print("ok  contraste de negro contra blanco da 21:1")


def test_un_gris_no_tiene_croma():
    assert croma("#808080") < 0.1
    print("ok  un gris puro da croma ≈ 0")


def test_un_color_no_se_separa_de_si_mismo():
    for daltonismo in [None, *MATRICES_DALTONISMO]:
        assert delta_e(COLOR_NUMERICA, COLOR_NUMERICA, daltonismo) < 1e-9
    print("ok  ΔE de un color contra sí mismo es 0 en las cuatro visiones")


def test_el_verde_intuitivo_falla_contra_el_naranja():
    assert delta_e("#2e8b57", COLOR_OBJETIVO) > PISO_DELTA_E
    assert delta_e("#2e8b57", COLOR_OBJETIVO, "protanopía") < PISO_DELTA_E
    print("ok  el validador rechaza el verde 'obvio': colapsa con el naranja en protanopía")


CATEGORICOS = {
    "numérica": COLOR_NUMERICA,
    "categórica": COLOR_CATEGORICA,
    "objetivo": COLOR_OBJETIVO,
}


def test_los_tres_colores_categoricos_se_separan_en_las_cuatro_visiones():
    nombres = sorted(CATEGORICOS)
    for i, uno in enumerate(nombres):
        for otro in nombres[i + 1:]:
            for daltonismo in [None, *MATRICES_DALTONISMO]:
                d = delta_e(CATEGORICOS[uno], CATEGORICOS[otro], daltonismo)
                assert d >= PISO_DELTA_E, (
                    f"{uno} vs {otro} en {daltonismo or 'visión normal'}: "
                    f"ΔE {d:.1f} < {PISO_DELTA_E}"
                )
    print("ok  los 3 pares categóricos superan el piso de ΔE en las 4 visiones")


def test_todos_los_colores_contrastan_contra_la_superficie():
    for nombre, color in CATEGORICOS.items():
        k = contraste(color, SUPERFICIE)
        assert k >= PISO_CONTRASTE, f"{nombre} ({color}): contraste {k:.2f} < {PISO_CONTRASTE}"
    print("ok  los 3 colores categóricos superan 3:1 contra la superficie")


def test_todos_los_colores_tienen_croma_suficiente():
    for nombre, color in CATEGORICOS.items():
        c = croma(color)
        assert c >= PISO_CROMA, f"{nombre} ({color}): croma {c:.1f} < {PISO_CROMA}"
    print("ok  los 3 colores categóricos superan el piso de croma")


def test_la_rampa_de_fumadores_es_monotona_en_luminosidad():
    claro, oscuro = luminancia(COLOR_FUMADOR_CLARO), luminancia(COLOR_FUMADOR_OSCURO)
    assert claro > oscuro, "la rampa de fumadores no es monótona"
    paso = contraste(COLOR_FUMADOR_CLARO, COLOR_FUMADOR_OSCURO)
    assert paso >= 2.0, f"el paso de la rampa es {paso:.2f}:1, demasiado chico"
    print(f"ok  la rampa de fumadores es monótona y su paso es {paso:.2f}:1")


def test_la_rampa_de_fumadores_se_distingue_del_azul():
    for color in (COLOR_FUMADOR_CLARO, COLOR_FUMADOR_OSCURO):
        for daltonismo in [None, *MATRICES_DALTONISMO]:
            d = delta_e(color, COLOR_NUMERICA, daltonismo)
            assert d >= PISO_DELTA_E, (
                f"{color} vs azul en {daltonismo or 'visión normal'}: ΔE {d:.1f}"
            )
    print("ok  los dos escalones de la rampa se separan del azul en las 4 visiones")


def main():
    test_negro_y_blanco_dan_el_contraste_maximo()
    test_un_gris_no_tiene_croma()
    test_un_color_no_se_separa_de_si_mismo()
    test_el_verde_intuitivo_falla_contra_el_naranja()

    test_los_tres_colores_categoricos_se_separan_en_las_cuatro_visiones()
    test_todos_los_colores_contrastan_contra_la_superficie()
    test_todos_los_colores_tienen_croma_suficiente()
    test_la_rampa_de_fumadores_es_monotona_en_luminosidad()
    test_la_rampa_de_fumadores_se_distingue_del_azul()

    print("TODOS LOS TESTS OK")


if __name__ == "__main__":
    main()
