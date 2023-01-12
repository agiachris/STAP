(define (problem structured_language_0)
	(:domain symbolic_workspace)
	(:objects
		hook - tool
		red_box - box
	)
	(:init
		(on hook table)
		(on red_box table)
	)
	(:goal (and
		(inhand red_box)
	))
)
