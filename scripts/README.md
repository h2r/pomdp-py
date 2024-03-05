# pomdp-py-release
Convenience for releasing pomdp_py


How to release pomdp-py:

1. For each python version you'd like to support, starting from low to high,
2. activate the virtulenv for that python version
3. Run `./release.sh <version>`

   (note that if pomdp-py repo is assumed to be at `$HOME/repo/pomdp-py`.

   This should generate both wheel and source builds,
   and place them under `pomdp-py/dist` properly.

4. After you've generated the wheels for all the versions of your interest,
   push to PyPI using twine:

    ```
    python3 -m twine upload --repository pypi dist/*
    ```

     ![image](https://github.com/zkytony/pomdp-py-release/assets/7720184/16f272bf-0996-464a-8678-34f3a150b890)


## Repository Convention (est. 07/25/23)

* When you're executing a release, besides the steps above, you should always:

   1. Merge the technical content of the `dev-<version>` branch into `master`.
   2. Create a separate PR with all the docs updates.

 This makes it so that the contributions of external collaborators are correctly counted by github. Otherwise, the Sphinx html doc builds will count thousands of lines of changes to the collaborators which are incorrect.
