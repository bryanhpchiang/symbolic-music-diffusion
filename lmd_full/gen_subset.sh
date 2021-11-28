find c  | head -n500 | tr '$\n' '\0' | xargs -0 -I {} cp {} subset/

