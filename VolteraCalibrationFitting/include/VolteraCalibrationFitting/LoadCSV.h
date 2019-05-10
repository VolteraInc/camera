/**
 * LoadCSV.
 *  
 * Ryan Wicks
 * 8 May 2019
 * Copyright Voltera Inc., 2019
 * 
 * Tool for loading variable length csv files.
 * 
 */
#ifndef LOAD_CSV_H
#define LOAD_CSV_H

#include <vector>
#include <string>

namespace voltera {

class LoadCSV {
public:
    /**
     * static function for loading csv files with variable line widths into memory as a vector of vectors. Anything that is not a number 
     * will be passed back as NaN.
     * 
     * @param filename file to open
     * @param loaded_data vector of vectors, one vector per line, data returned by reference.
     * @return was the load successful.
     */
    static bool load ( const std::string & filename, std::vector < std::vector <double> > & loaded_data );

private:

    ///< string of characters that denote comments
    static std::string c_comment_delimiters;

    ///< delimiter char of characters that denote comments
    static char c_delimiter;
};

}

#endif //LOAD_CSV_H