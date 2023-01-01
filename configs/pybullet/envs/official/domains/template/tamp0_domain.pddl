(define (domain workspace)
	(:requirements :strips :typing :equality :universal-preconditions :negative-preconditions :conditional-effects)
	(:types
		physobj - object
		unmovable - physobj
		movable - physobj
		tool - movable
		box - movable
		rack - rack
	)
	(:constants table - unmovable)
	(:predicates
		(inhand ?a - movable)
		(on ?a - movable ?b - physobj)
		(inworkspace ?a - movable)
		(beyondworkspace ?a - physobj)
		(under ?a - movable ?b - unmovable)
	)
	(:action pick
		:parameters (?a - movable)
		:precondition (and
			(exists (?b - physobj) (on ?a ?b))
			(forall (?b - movable)
			(and
				(not (inhand ?b))
				(not (on ?b ?a))
			)
			)
		)
		:effect (and
			(inhand ?a)
			(forall (?b - physobj) (not (on ?a ?b)))
		)
	)
	(:action place
		:parameters (?a - movable ?b - unmovable)
		:precondition (and
			(not (= ?a ?b))
			(inhand ?a)
		)
		:effect (and
			(not (inhand ?a))
			(on ?a ?b)
		)
	)
	(:action pull
		:parameters (?a - box ?b - tool)
		:precondition (and
			(not (= ?a ?b))
			(inhand ?b)
			(on ?a table)
		)
		:effect (and
			(inworkspace ?a)
		)
	)
    (:action push
        :parameters (?obj - movable ?tool - tool ?dest - rack)
        :precondition (and
            (inhand ?tool)
            (on ?obj table)
            (not (under ?obj ?rack))
        )
        :effect (and
            (under ?obj ?dest)
        )
    )
)
