#!/bin/tcsh -f

foreach fig ( figure_* )
    cd $fig
    tcsh ${fig}.tcsh
    cd ..
end
