(define (problem structured_language_0)
	(:domain symbolic_workspace)
	(:objects
		rack - receptacle
		hook - tool
		red_box - box
	)
	(:init
		(on rack table)
		(on hook table)
		(on red_box table)
	)
	(:goal (and
		(inhand red_box)
	))
)
