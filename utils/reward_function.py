# On a utilisé la même signature de fonction afin de pouvoir généraliser l'appel sans soucis

def instant_reward1 (is_crash, game_reward, coord_variation) :
    """ Fonction reward V1"""
    score = 0
    if is_crash :
        score = -100
    elif coord_variation < -100 :
        score = 100 * game_reward
    else :
        score = 2 * coord_variation
    return score

def instant_reward2 (is_crash, game_reward, coord_variation) :
    """ Fonction reward V2"""
    score = 0
    if is_crash :
        score = -100
    elif coord_variation < -100 :
        score = 100
    else :
        if coord_variation > 0 :
            score = 5
        elif coord_variation == 0 :
            score = 1
        else :
            score = -10
    return score

def instant_reward3 (is_crash, game_reward, coord_variation) :
    """ Fonction reward simpliste"""
    score = 0
    if coord_variation > 0 :
        score = 0.5
    elif coord_variation == 0 :
        score = -0.01
    return score