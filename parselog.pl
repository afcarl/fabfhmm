#!/usr/bin/env perl
use strict;
use warnings;
use List::Util qw(sum min max);
use Statistics::Descriptive;
use File::Basename;
use Getopt::Std;

my $OPTIONS = "n:s:";
my %opt;
getopts($OPTIONS, \%opt);

if( @ARGV==0 ){
	die "Usage: $0 log_dir\n";
}

my $selTrial;
if( exists $opt{'s'} ){
    $selTrial = $opt{'s'};
}

my ($trainSeqNum, $testSeqNum);

if( exists $opt{'n'} ){
    ($trainSeqNum, $testSeqNum) = split /,/, $opt{'n'};
}
else{
    ($trainSeqNum, $testSeqNum) = (1, 1);
}

$| = 1;

my @filenames;
my @filetuples;
my $FH;
my $basedir;

if( -d $ARGV[0] ){
	my $dh;
	opendir($dh, $ARGV[0]) || die "can't opendir $ARGV[0]: $!\n";
	
	@filenames = readdir($dh);
	@filenames = grep { /^(CFAB|FAB|ML)-(\d+)-[^.]+\.log/ } @filenames;
	$basedir = $ARGV[0];
}
elsif( -f $ARGV[0] ){
	$basedir = dirname( $ARGV[0] );
	my $filename = basename( $ARGV[0] );
	@filenames = ( $filename );
}
@filetuples = map { if( /^(CFAB|FAB|ML)-(\d+)-(\d+)-[^.]+\.log/ ){ [ $3, $1, $2, $_ ]; } } @filenames;
@filetuples = sort { $a->[0] <=> $b->[0] || $a->[1] cmp $b->[1] || $a->[2] <=> $b->[2] } @filetuples;
@filetuples = map { [ $_->[0], $_->[1] . '-' . $_->[2], $_->[3] ] } @filetuples;

my %stats;
my $verbose = 1;

for my $filetuple( @filetuples ){
	my ($trial, $type, $filename) = @$filetuple;
	
	#last if $trial > 10;
	
	open($FH, "< $basedir/$filename") || die "cannot open $ARGV[0]$filename: $!\n";
	my $line;
	my ($tlfab, $tlfabavg, $plfab, $plfabavg, $plml, $plmlavg, $tlml, $tlmlavg);
	my ($iter, $fic, $statenum);
	
	$stats{$type}{$trial}{iters} = [];
	$stats{$type}{$trial}{test} = {};
	
	my $testvstep = -1;
	$iter = -1;
	my $lastPartialFIC = -1;
	
	while( $line = <$FH> ){
		if( $line =~ /^(\d+) FIC:  ([\d.-]+) \[((\d+, )*\d+)\]/ ){
			$iter = $1;
			$fic = $2;
			my $statenumStr = $3;
			my @statenums = split /, /, $statenumStr;
			@statenums = sort { $a <=> $b } @statenums;
			my $statenumSum = sum(@statenums);
			$stats{$type}{$trial}{iters}[$iter] = 
					{ fic => $fic, statenumSum => $statenumSum, statenums => \@statenums };
			next;
		}
		if( $line =~ /E_q \[ log p\(x,z\|theta\) \]: ([\d.-]+)/ ){
			if( $testvstep == -1 && $iter >= 0 ){
				$stats{$type}{$trial}{iters}[$iter]{Eq_pxz} = $1;
			}
			else{
				push @{ $stats{$type}{$trial}{test}{Eq_pxz} }, $1;
			}
			next;
		}
		if( $line =~ /Partial FIC: ([\d.-]+)/ ){
			$lastPartialFIC = $1;
			if( $lastPartialFIC =~ /\.$/ ){
				$lastPartialFIC = substr($lastPartialFIC, 0, -1);
			}
			next;
		}
		if( $line =~ /(FAB|ML) negKL: ([\d.-]+)/ ){
			my $negkl = $2;
			if( $negkl =~ /\.$/ ){
				$negkl = substr($negkl, 0, -1);
			}
			if( $testvstep >= 0 ){
				push @{ $stats{$type}{$trial}{test}{negkl} }, $negkl;
			}
			next;
		}
		if( $line =~ /^Training sequences log likelihood: ([\d.-]+). Avg: ([\d.-]+)/ ){
			$stats{$type}{$trial}{trainlike} = $1;
			$stats{$type}{$trial}{avgtrainlike} = $2;
			next;
		}
		if( $line =~ /Test sequence V-step (\d+)/ ){
			$testvstep = $1 - 1;
		}
	}
	
	# empty log file
	if($iter == -1 || $testvstep == -1){
		next;
	}
	
	my $lastIdx = -1;
	#if( $type eq "ML" ){
	#	$lastIdx = 20;
	#}
	my $sumK = $stats{$type}{$trial}{iters}[$lastIdx]{statenumSum};
	my $Ks = $stats{$type}{$trial}{iters}[$lastIdx]{statenums};
	print "$filename: ";
		
	#my $bestTestLike = max( @{$stats{$type}{$trial}{test}} );
	#my $avgTestlike = $bestTestLike / 2000;
	my $bestTestNegKL = max( @{ $stats{$type}{$trial}{test}{negkl} } );
	my $bestTestLike = $bestTestNegKL + $lastPartialFIC;
	my $avgTestlike = $bestTestLike / 2000;
	
	my $trainlike = $stats{$type}{$trial}{iters}[-1]{fic};
	my $avgTrainlike = $trainlike / 2000;
	#my $finalFIC = $stats{$type}{$trial}{iters}[-1]{fic};
	$stats{$type}{$trial}{Ks} = $Ks;
	$stats{$type}{$trial}{testlike} = $bestTestLike;
	$stats{$type}{$trial}{trainlike} = $trainlike;
	
	if( $verbose ){
		print "$iter iters, state num @$Ks. Trainlike $trainlike, avg $avgTrainlike. ",
				"Testlike $bestTestLike, avg $avgTestlike\n";
	}
}
print "\n";

if( $selTrial ){
	for my $type( sort keys %stats ){
		#print "$type-$selTrial:\n";
		my @statenumSums = map { $_->{statenumSum} } @{ $stats{$type}{$selTrial}{iters} };
		if( @statenumSums > 400 ){
			@statenumSums = @statenumSums[0..399];
		}
		my $step = 10;
		
		#if($type eq "ML"){
		#	$step = 1;
		#	@statenumSums = @statenumSums[0..39];
		#}
		print "${type}_ky = [ ...\n";
		my $i;
		for($i = 0; $i*$step < @statenumSums; $i++){
			my $j = $i*$step;
			print "$statenumSums[$j],";
			if( $i > 0 && $i %15 == 14 ){
				print "...\n";
			}
		}
		print " ];\n";
		#if($type eq "ML"){
		#	$step = 10;
		#}
		my $j = ($i - 1)*$step;
		print "${type}_kx = [ 0:$step:$j ];\n";
		
		$step = 10;
		my @trainlikes = map { $_->{fic} } @{ $stats{$type}{$selTrial}{iters} };
		if( @trainlikes > 150 ){
			@trainlikes = @trainlikes[0..149];
		}
		
		print "${type}_ly = [ ...\n";
		for($i = 0; $i*$step < @trainlikes; $i++){
			my $j = $i*$step;
			print "$trainlikes[$j],";
			if( $i > 0 && $i %5 == 4 ){
				print "...\n";
			}
		}
		print " ];\n";
		$j = ($i - 1)*$step;
		print "${type}_lx = [ 0:$step:$j ];\n";		
	}
	exit;
}

for my $type( sort keys %stats ){
	my $trialNum = scalar keys %{ $stats{$type} };
	print "$trialNum $type logs read\n";
	
	my @allKs = map{ $stats{$type}{$_}->{Ks} } keys %{ $stats{$type} };
	@allKs = grep { defined } @allKs;
	if( @allKs == 0 ){
		print "All empty.\n";
		next;
	}
	
	my $layercount = @{ $allKs[0] };
	my $calc = Statistics::Descriptive::Full->new();
	my $i;
	printf "Ks: ";
	
	for($i = 0; $i < $layercount; $i++){
		my @allK = map{ $_->[$i] } @allKs;
		
		$calc->add_data(@allK);
		my $avgK = $calc->mean();
		my $divK = $calc->standard_deviation();
		printf "%.1f (%.1f), ", $avgK, $divK;
		$calc->clear();
	}
	print "\n";
	
	my @objectives = ( "trainlike", "testlike" );
	my @seqNums = ( $trainSeqNum, $testSeqNum );
	
	for  (my $i = 0; $i < 2; $i++ ){
	    my $obj = $objectives[$i];
	    my $seqNum = $seqNums[$i];
	    
		my @scores = map{ $stats{$type}{$_}{$obj} } keys %{ $stats{$type} };
		@scores = grep { defined } @scores;
		if( @scores == 0 ){
			print "$obj empty.\n";
			next;
		}
		$calc->add_data(@scores);
		my $avg = $calc->mean();
		my $div = $calc->standard_deviation();
		$calc->clear();
		
		printf "$obj all seqs: %.1f (%.1f)\n", $avg, $div;
		printf "$obj each seq: %.1f (%.1f)\n", $avg / $seqNum, $div / $seqNum;
	}
	print "\n";
}
