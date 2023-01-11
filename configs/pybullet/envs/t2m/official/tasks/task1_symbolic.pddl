(define (problem structured_language_1)
	(:domain symbolic_workspace)
	(:objects
        rack - receptacle
		hook - tool
		red_box - box
        yellow_box - box
        cyan_box - box
        blue_box - box
	)
	(:init
		(on rack table)
        (on hook rack)
        (on red_box table)
		(on yellow_box table)
		(on cyan_box table)
		(on blue_box table)
	)
	(:goal (and
		(on hook table)
	))
)
