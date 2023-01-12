(define (problem lifted_1)
	(:domain geometric_workspace)
	(:objects
        rack - receptacle
        hook - tool
        cyan_box - box
        blue_box - box
	)
	(:init
		(on rack table)
        (on hook table)
		(on cyan_box table)
		(on blue_box table)
        ; Geometric facts
        (inworkspace table)
        (inworkspace rack)
        (inworkspace hook)
        (inworkspace cyan_box)
        (beyondworkspace blue_box)
        (nonblocking blue_box rack)
        (nonblocking blue_box cyan_box)
	)
	(:goal (and
        (on blue_box rack)
    ))
)
