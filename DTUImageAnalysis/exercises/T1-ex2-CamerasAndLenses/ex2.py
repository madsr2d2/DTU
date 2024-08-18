import math


def camera_b_distance(f, g):
    """
    Calculate the image distance `b` using the Gaussian lens equation.

    The Gaussian lens equation relates the focal length `f` of a lens, the distance `g`
    from the object to the lens (object distance), and the distance `b` from the lens
    to the image formed (image distance). The equation is given by:

        1/f = 1/g + 1/b

    This function rearranges the equation to solve for `b` (the image distance):

        b = 1 / (1/f - 1/g)

    """
    return 1 / (1 / f - 1 / g)


g_distances = (100, 1000, 5000, 15000)

for g in g_distances:
    b = round(camera_b_distance(f=15, g=g), 1)
    print(
        f"Focal length (f) = 15 mm\nObject distance (g) = {g} mm\nImage distance (b) = {b} mm\n"
    )
