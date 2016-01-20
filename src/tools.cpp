
#include "tools.h"

#include <stdio.h>
#include <string>
#include <string.h>
#include <sys/time.h> // for gettimeofday
#include <time.h> // for gettimeofday

//---------------------------------------------------------

/**
	Return elapsed time since last call, in milli-seconds.
	Manage 10 counters.
*/
double top(int id)
{
#ifndef WIN32
	/* linux */
	struct timeval tv; 
	static double previousTime[10] = { 0.0 };
	double delta_t;

	// read current time
	gettimeofday( &tv, NULL);

	// compute elapsed time since last call
	double time = tv.tv_sec*1000.0 + tv.tv_usec*0.001;
	delta_t = time - previousTime[id];

	// save current time for next call
	previousTime[id] = time;

	// return elapsed time in ms
	return delta_t;
#else
	/* windows */
	DWORD newT;
	static DWORD oldT[10] = { 0 };
	double delta_t;

	// read current time in ms
	newT = GetTickCount();

	// compute elapsed time since last call, in ms  
	delta_t = newT - oldT[id];

	// save current time for next call
	oldT[id] = newT;

	// return elapsed time in ms
	return delta_t;
#endif
}

//---------------------------------------------------------

/**
	Get argument value from string like "argname=value".
*/
const char *getArg( const char *str, const char *argName)
{
	if( strstr( str, argName) != str )
		return NULL; // not found
		
	return strchr( str, '=') + 1;
}

//---------------------------------------------------------

/**
	Get argument value from command line.
*/
const char *getArgValueFromCmdl( const char _argc, const char **_argv, 
	const char *argName)
{
	const char *argVal = NULL;
	std::string pattern(argName);
	pattern.append("=");

	for( int iArg = 1; iArg < _argc; iArg++)
	{
		if( (argVal = getArg( _argv[iArg], pattern.c_str())) != NULL )
			break;
	}
	
	return argVal;
}

//---------------------------------------------------------

