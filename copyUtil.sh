#!/bin/bash
echo "start moving data..."
declare -a arr=("kgs-19-2008" "kgs-19-2009" "kgs-19-2010" "kgs-19-2011" "kgs-19-2012" "kgs-19-2013" "kgs-19-2014" "kgs-19-2015" "kgs-19-2016-01-new" "kgs-19-2016-02-new" "kgs-19-2016-03-new" "kgs-19-2016-04-new" "kgs-19-2016-05-new" "kgs-19-2016-06-new" "kgs-19-2016-07-new" "kgs-19-2016-08-new" "kgs-19-2016-09-new" "kgs-19-2016-10-new" "kgs-19-2016-11-new" "kgs-19-2016-12-new" "kgs-19-2017-01-new" "kgs-19-2017-02-new" "kgs-19-2017-03-new" "kgs-19-2017-04-new" "kgs-19-2017-05-new" "kgs-19-2017-06-new" "kgs-19-2017-07-new" "kgs-19-2017-08-new" "kgs-19-2017-09-new" "kgs-19-2017-10-new" "kgs-19-2017-11-new" "kgs-19-2017-12-new" "kgs-19-2018-01-new" "kgs-19-2018-02-new" "kgs-19-2018-03-new" "kgs-19-2018-04-new" "kgs-19-2018-05-new" "kgs-19-2018-06-new" "kgs4d-19-2007" "kgs4d-19-2008" "kgs4d-19-2009" "kgs4d-19-2010" "kgs4d-19-2011" "kgs4d-19-2012" "kgs4d-19-2013" "kgs4d-19-2014" "kgs4d-19-2015" "kgs_data2")
for i in "${arr[@]}"
do
    echo "$i"
    find "$i" -name "*.sgf" -exec mv {} kgs_data/ \;
    ls "$i"
    rm -rf "$i"
done
echo "Done"
